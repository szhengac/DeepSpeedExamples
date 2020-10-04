#! /bin/bash

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

hostfile="$script_dir/hostfile"

config_json="$script_dir/ds_bert_config.json"
bert_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 16 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 500000 \
       --save checkpoints/bert_345m \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --log-interval 100
"

mpi_options=" \
	--allow-run-as-root -np 16 -N ${NUM_GPUS_PER_WORKER} --hostfile ${hostfile} \
	--mca pml ob1 --mca btl ^openib \
       	--mca btl_tcp_if_exclude docker0,lo \
	--mca plm_rsh_num_concurrent 300 --mca routed_radix 600 \
	--bind-to none \
	-x NCCL_SOCKET_IFNAME=eth0 \
	-x NCCL_IB_HCA=eth0 \
	-x FI_PROVIDER='efa' \
       	-x FI_EFA_TX_MIN_CREDITS=64 \
	-x FI_OFI_RXR_RX_COPY_UNEXP=1 -x FI_OFI_RXR_RX_COPY_OOO=1 -x FI_EFA_MR_CACHE_ENABLE=1 -x FI_OFI_RXR_INLINE_MR_ENABLE=1 \
	-x LD_LIBRARY_PATH=$HOME/aws-ofi-nccl/install/lib/:$HOME/nccl/build/lib:/usr/local/cuda-10.0/lib64:/opt/amazon/efa/lib64:$LD_LIBRARY_PATH \
	-x NCCL_MIN_NRINGS=1 \
	-x NCCL_DEBUG=VERSION \
	-x NCCL_TREE_THRESHOLD=4294967296
"

run_cmd="mpirun ${mpi_options} python3 pretrain_bert.py $@ ${bert_options} --deepspeed_mpi --deepspeed --deepspeed_config ${config_json}"
echo ${run_cmd}
eval ${run_cmd}

set +x
