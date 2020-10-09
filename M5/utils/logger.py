import logging
import torch.distributed as dist

class Logger():
    def __init__(self, cuda=False, filename=None, level=logging.INFO, filemode='w'):
        logging.basicConfig(
                filename=filename,
                filemode=filemode,
                format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                level=level)
        self.logger = logging.getLogger(__name__)
        self.cuda = cuda

    def info(self, message, *args, **kwargs):
        if (self.cuda and dist.get_rank() == 0) or not self.cuda:
            self.logger.info(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)
