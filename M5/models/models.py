import torch

from turing.utils import TorchTuple

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from .bert import BertModel, BertForPreTrainingPreLN, BertConfig


class BertMultiTask:
    def __init__(self, args):
        self.config = args.config

        if not args.use_pretrain:

            bert_config = BertConfig(**self.config["bert_model_config"])
            bert_config.vocab_size = len(args.tokenizer.vocab)

            # Padding for divisibility by 8
            if bert_config.vocab_size % 8 != 0:
                bert_config.vocab_size += 8 - (bert_config.vocab_size % 8)
            print("VOCAB SIZE:", bert_config.vocab_size)

            self.network = BertForPreTrainingPreLN(bert_config, args)
        # Use pretrained bert weights
        else:
            self.bert_encoder = BertModel.from_pretrained(
                self.config['bert_model_file'],
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
                'distributed_{}'.format(args.local_rank))

        self.device = None

    def set_device(self, device):
        self.device = device

    def save(self, filename: str):
        network = self.network.module
        return torch.save(network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.module.load_state_dict(
            torch.load(model_state_dict,
                       map_location=lambda storage, loc: storage))

    def move_batch(self, batch: TorchTuple, non_blocking=False):
        return batch.to(self.device, non_blocking)

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save_bert(self, filename: str):
        return torch.save(self.bert_encoder.state_dict(), filename)

    def to(self, device):
        assert isinstance(device, torch.device)
        self.network.to(device)

    def half(self):
        self.network.half()
