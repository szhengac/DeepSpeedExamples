import os
import sys
import time
import logging
import numpy as np
import random
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.logger import Logger
from utils.args import get_argument_parser
from utils.utils import get_sample_writer, is_time_to_exit
from models.mpu_models import BertMultiTask
from datasets.dataset import PreTrainingDataset, PretrainDataType
from tokenization.tokenization import BertTokenizer
from optimization.optimization import warmup_linear_decay_exp, warmup_exp_decay_exp, warmup_exp_decay_poly, warmup_linear_const_decay_poly

from datasets.nvidia_bert_dataset_provider import NvidiaBertDatasetProvider

import mpu
import deepspeed

global_step = 0
global_data_samples = 0
last_global_step_from_restore = 0

keys = ['BatchType', 'input_ids', 'mask', 'masked_lm_labels']
datatype = torch.int64


def checkpoint_model(PATH, ckpt_id, model, last_global_step,
                     last_global_data_samples, **kwargs):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    checkpoint_state_dict = {
        'last_global_step': last_global_step,
        'last_global_data_samples': last_global_data_samples
    }
    # Add extra kwargs too
    checkpoint_state_dict.update(kwargs)

    success = model.network.save_checkpoint(PATH, ckpt_id,
                                            checkpoint_state_dict)
    status_msg = 'checkpointing: PATH={}, ckpt_id={}'.format(PATH, ckpt_id)
    if success:
        logging.info(f"Success {status_msg}")
    else:
        logging.warning(f"Failure {status_msg}")
    return


def load_training_checkpoint(args, model, PATH, ckpt_id):
    """Utility function for checkpointing model + optimizer dictionaries
       The main purpose for this is to be able to resume training from that instant again
    """
    logger = args.logger
    _, checkpoint_state_dict = model.network.load_checkpoint(PATH, ckpt_id)
    last_global_step = checkpoint_state_dict['last_global_step']
    last_global_data_samples = checkpoint_state_dict[
        'last_global_data_samples']
    del checkpoint_state_dict
    return last_global_step, last_global_data_samples


def get_dataloader(args, dataset: Dataset, eval_set=False):
    if args.local_rank == -1:
        train_sampler = RandomSampler(dataset)
    else:
        train_sampler = DistributedSampler(dataset,
                                           num_replicas=mpu.get_data_parallel_world_size(),
                                           rank=mpu.get_data_parallel_rank())
    return (x for x in
            DataLoader(dataset,
                       batch_size=args.train_micro_batch_size_per_gpu //
                       2 if eval_set else args.train_micro_batch_size_per_gpu,
                       sampler=train_sampler,
                       num_workers=args.config['training']['num_workers']))


def pretrain_validation(args, index, model):
    global global_step

    if args.validation_data_path_prefix is None:
        return

    config = args.config
    logger = args.logger

    logger.info(
        f"Validation micro batch size: {args.train_micro_batch_size_per_gpu}")

    model.eval()
    dataset = PreTrainingDataset(
        args.tokenizer,
        os.path.join(args.validation_data_path_prefix,
                     config['validation']['path']), args.logger,
        args.max_seq_length, index, PretrainDataType.VALIDATION,
        args.max_predictions_per_seq)
    if mpu.get_model_parallel_rank() == 0:
        data_batches = get_dataloader(args, dataset, eval_set=True)
        num_iterations = torch.cuda.LongTensor([len(data_batches)])
    else:
        num_iterations = torch.cuda.LongTensor([0])
    torch.distributed.broadcast(num_iterations,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    num_iterations = num_iterations[0].item()
    eval_loss = 0
    nb_eval_steps = 0
    for i in tqdm(range(num_iterations)):
        # broadcast batch from src model parallel rank to others
        if mpu.get_model_parallel_rank() == 0:
            batch = data_batches[i]
            data = {k: v for k, v in zip(keys, batch)}
        else:
            data = None
        data = mpu.broadcast_data(keys, data, datatype)
        batch = [data[k] for k in keys]

        batch = tuple(t.to(args.device) for t in batch)
        tmp_eval_loss = model.network(batch, log=False)
        dist.reduce(tmp_eval_loss, 0)
        # Reduce to get the loss from all the GPU's
        tmp_eval_loss = tmp_eval_loss / dist.get_world_size()
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    logger.info(f"Validation Loss for global step {global_step} is: {eval_loss}")
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Validation/Loss', eval_loss,
                                       index + 1)
    return


def master_process(args):
    return (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1)


def train(args,
          index,
          model,
          optimizer,
          pretrain_dataset_provider,
          finetune=False):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    dataset_iterator, total_length = pretrain_dataset_provider.get_shard(index)
    current_data_sample_count = global_data_samples

    config = args.config
    logger = args.logger
    logger.info(
        f'worker-{dist.get_rank()}: begin {index+1}-th shard current_sample_count {current_data_sample_count} shard_length {total_length} global_data_samples {global_data_samples}'
    )

    if pretrain_dataset_provider is not None:
        pretrain_dataset_provider.prefetch_shard(index + 1)

    model.train()

    rounds = 20
    all_step_time = 0.0
    step_counts = 0

    for _, batch_index in enumerate(tqdm(dataset_iterator, smoothing=1)):
        try:
            step_start = time.time()

            # broadcast batch from src model parallel rank to others
            if pretrain_dataset_provider is not None:
                batch = pretrain_dataset_provider.get_batch(batch_index)
                data = {k: v for k, v in zip(keys, batch)}
            else:
                data = None
            data = mpu.broadcast_data(keys, data, datatype)
            batch = [data[k] for k in keys]

            batch = tuple(t.to(args.device) for t in batch)  # Move to GPU

            # Calculate forward pass
            loss = model.network(batch)
            unscaled_loss = loss.item()
            current_data_sample_count += (args.train_micro_batch_size_per_gpu *
                                          dist.get_world_size())

            if pretrain_dataset_provider is not None:
                # Prefetch training data
                pretrain_dataset_provider.prefetch_batch()

            model.network.backward(loss)

            loss = None

            if model.network.is_gradient_accumulation_boundary():
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = update_learning_rate(
                        args, config, global_step, optimizer)

                    report_step_metrics(args, lr_this_step, unscaled_loss,
                                        global_step, current_data_sample_count)

                model.network.step()

                report_lamb_coefficients(args, optimizer)
                global_step += 1

                if global_step % args.save_ckpt_interval == 0:
                    logger.info(f"Saving a checkpointing of the model for step: {global_step}")
        
                    checkpoint_model(PATH=args.saved_model_path,
                                    ckpt_id='global_step_{}'.format(global_step),
                                    model=model,
                                    last_global_step=global_step,
                                    last_global_data_samples=global_data_samples)

                # Run Validation Loss
                if global_step % args.validation_interval == 0 and not finetune:
                    pretrain_validation(args, index, model)
            else:
                # Call DeepSpeed engine step on micro steps
                model.network.step()

        except StopIteration:
            continue

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args,
                           global_steps=current_global_step):
            logging.warning(
                f'Termination due to max steps limit, global step = {current_global_step}'
            )
            break
        step_time = time.time() - step_start
        all_step_time += step_time
        if global_step % rounds == 0 and global_step != 0 and model.network.is_gradient_accumulation_boundary(
        ) and dist.get_rank() == 0:
            one_step_bs = args.train_micro_batch_size_per_gpu * args.gradient_accumulation_steps * dist.get_world_size(
            ) * rounds
            logger.info(' At step {}, the throughput is {:2f} Samples/s'.format(
                global_step * args.gradient_accumulation_steps,
                one_step_bs / all_step_time))
            all_step_time = 0.0

    if pretrain_dataset_provider is not None:
        pretrain_dataset_provider.release_shard(index)

    global_data_samples = current_data_sample_count


def update_learning_rate(args, config, current_global_step, optimizer):
    global last_global_step_from_restore

    global_step_for_lr = current_global_step - last_global_step_from_restore

    if args.lr_schedule == "EE":
        #logging.info(f'LR Schedule is {args.lr_schedule} EE')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == "EP":
        #logging.info(f'LR Schedule is {args.lr_schedule} EP')
        lr_this_step = config["training"][
            "learning_rate"] * warmup_exp_decay_poly(
                global_step_for_lr, config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    elif args.lr_schedule == 'LP':
        lr_this_step = config["training"][
            "learning_rate"] * warmup_linear_const_decay_poly(
                global_step_for_lr, config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"], config["training"]["const_proportion"])
    elif args.lr_schedule == 'LE':
        lr_this_step = config["training"][
            "learning_rate"] * warmup_linear_decay_exp(
                global_step_for_lr, config["training"]["decay_rate"],
                config["training"]["decay_step"],
                config["training"]["total_training_steps"],
                config["training"]["warmup_proportion"])
    else:
        raise NotImplementedError
    lr_this_step += args.lr_offset

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_this_step

    return lr_this_step


def report_step_metrics(args, lr, loss, step, data_sample_count):
    ##### Record the LR against global_step on tensorboard #####
    logger = args.logger
    if (not args.no_cuda
            and dist.get_rank() == 0) or (args.no_cuda
                                          and args.local_rank == -1):
        args.summary_writer.add_scalar(f'Train/lr', lr, step)

        args.summary_writer.add_scalar(f'Train/Samples/train_loss', loss,
                                       data_sample_count)

        args.summary_writer.add_scalar(f'Train/Samples/lr', lr,
                                       data_sample_count)
    ##### Recording  done. #####

    if (step + 1) % args.print_steps == 0 and master_process(args):
        logger.info('bert_progress: step={}, loss={}, lr={}, sample_count={}'.
              format(step + 1, loss, lr, data_sample_count))


def report_lamb_coefficients(args, optimizer):
    logger = args.logger
    if master_process(args):
        if (args.fp16 and args.use_lamb):
            logger.info("Lamb Coeffs: {}".format(optimizer.optimizer.get_lamb_coeffs()))
            lamb_coeffs = optimizer.optimizer.get_lamb_coeffs()
            lamb_coeffs = np.array(lamb_coeffs)
            if lamb_coeffs.size > 0:
                args.summary_writer.add_histogram(f'Train/lamb_coeffs',
                                                  lamb_coeffs, global_step)


def get_arguments():
    parser = get_argument_parser()
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    # no cuda mode is not supported
    args.no_cuda = False

    return args


def construct_arguments():
    args = get_arguments()

    # Prepare Logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = Logger(cuda=torch.cuda.is_available() and not args.no_cuda,
                    filename=args.log_file, level=log_level)
    args.logger = logger
    config = json.load(open(args.config_file, 'r', encoding='utf-8'))

    # choose dataset and training config based on the given sequence length
    seq_len = str(args.max_seq_length)

    datasets = config["data"]["mixed_seq_datasets"][seq_len]
    del config["data"]["mixed_seq_datasets"]
    training = config["mixed_seq_training"][seq_len]
    del config["mixed_seq_training"]
    config["data"]["datasets"] = datasets
    config["training"] = training
    args.config = config
    args.max_steps = config["training"]["total_training_steps"]

    args.job_name = config['name'] if args.job_name is None else args.job_name
    logging.info("Running Config File: {}".format(args.job_name))
    # Setting the distributed variables
    logging.info("Args = {}".format(args))

    # Setting all the seeds so that the task is random but same accross processes
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    args.saved_model_path = os.path.join(args.output_dir, "saved_models/",
                                         args.job_name)

    args.n_gpu = 1

    # Loading Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    # Set validation dataset path
    if args.validation_data_path_prefix is None:
        logging.warning(
            'Skipping validation because validation_data_path_prefix is unspecified'
        )

    logging.info('Training exit is set after {} global steps'.format(args.max_steps))

    return args


def prepare_optimizer_parameters(args, model):
    config = args.config

    param_optimizer = list(model.network.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if "weight_decay" in config["training"].keys():
        weight_decay = config["training"]["weight_decay"]
    else:
        weight_decay = 0.01

    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    return optimizer_grouped_parameters


def prepare_model_optimizer(args):
    # Initialize torch distributed
    # torch.distributed.init_process_group(backend="nccl")

    # Loading Model
    model = BertMultiTask(args)

    if mpu.get_data_parallel_rank() == 0:
        logging.info(' Number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # Optimizer parameters
    optimizer_grouped_parameters = prepare_optimizer_parameters(args, model)

    # DeepSpeed initializer handles FP16, distributed, optimizer automatically.
    model.network, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model.network,
        model_parameters=optimizer_grouped_parameters,
        mpu=mpu,
        dist_init_required=False)

    # Overwrite application configs with DeepSpeed config
    args.train_micro_batch_size_per_gpu = model.network.train_micro_batch_size_per_gpu(
    )
    args.gradient_accumulation_steps = model.network.gradient_accumulation_steps(
    )

    # Set DeepSpeed info
    args.local_rank = model.network.local_rank
    args.device = model.network.device
    model.set_device(args.device)
    args.fp16 = model.network.fp16_enabled()
    args.use_lamb = model.network.optimizer_name(
    ) == deepspeed.runtime.config.LAMB_OPTIMIZER

    # Prepare Summary Writer and saved_models path
    if dist.get_rank() == 0:
        summary_writer = get_sample_writer(name=args.job_name,
                                           base=args.output_dir)
        args.summary_writer = summary_writer
        os.makedirs(args.saved_model_path, exist_ok=True)

    return model, optimizer


def load_checkpoint(args, model):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    logger.info(
        f"Restoring previous training checkpoint from PATH={args.load_training_checkpoint}, CKPT_ID={args.load_checkpoint_id}"
    )
    global_step, global_data_samples = load_training_checkpoint(
        args=args,
        model=model,
        PATH=args.load_training_checkpoint,
        ckpt_id=args.load_checkpoint_id)
    logger.info(
        f"The model is loaded from last checkpoint when the global steps were at {global_step} and global data samples at {global_data_samples}"
    )

    if args.rewarmup:
        logger.info(
            f"Rewarmup learning rate with last_global_step_from_restore = {global_step}"
        )
        last_global_step_from_restore = global_step

    lr_this_step = config["training"][
        "learning_rate"] * warmup_linear_decay_exp(
            global_step, config["training"]["decay_rate"],
            config["training"]["decay_step"],
            config["training"]["total_training_steps"],
            config["training"]["warmup_proportion"])
    logger.info(f"Restart training with lr = {lr_this_step}")


def run(args, model, optimizer):
    global global_step
    global global_data_samples
    global last_global_step_from_restore

    config = args.config
    logger = args.logger

    # Only the first model parallel rank needs to load data
    pretrain_dataset_provider = None
    if mpu.get_model_parallel_rank() == 0:
        pretrain_dataset_provider = NvidiaBertDatasetProvider(args, mpu=mpu)

    index = 0
    while True:
        logger.info(f"Training {index + 1}-th shard")
        pre = time.time()
        train(args, index, model, optimizer, pretrain_dataset_provider)
        post = time.time()
        logger.info(f"Time for {index + 1}-th shard: {post-pre} seconds")

        current_global_step = global_step - last_global_step_from_restore
        if is_time_to_exit(args=args, global_steps=current_global_step):
            logger.info(
                f'Warning: Early training termination due to max steps limit, global_step={current_global_step}'
            )
            break

        index += 1


'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''
def set_deepspeed_activation_checkpointing(args):
    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config,
                                      num_checkpoints=args.config["bert_model_config"]['num_hidden_layers'])
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_mpu(args):
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)

    mpu.model_parallel_cuda_manual_seed(args.seed)


def main():
    start = time.time()
    args = construct_arguments()
    initialize_mpu(args)
    model, optimizer = prepare_model_optimizer(args)
    if not None in [args.load_training_checkpoint, args.load_checkpoint_id]:
        load_checkpoint(args, model)
    run(args, model, optimizer)
    elapsed = time.time() - start
    logger = args.logger
    logger.info(f"Elapsed time: {elapsed} seconds")


if __name__ == "__main__":
    main()
