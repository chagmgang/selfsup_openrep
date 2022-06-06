import os
import time

from selfsup.logger import LOGGERS
from selfsup.utils import get_root_logger
from .base import BaseLogger
from .utils import rank_zero_only


@LOGGERS.register_module()
class PrintLogger(BaseLogger):

    def __init__(self, interval=5, **kwargs):
        super(PrintLogger, self).__init__()

        self.interval = interval

    @rank_zero_only
    def before_run(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.work_dir, f'{timestamp}.log')
        self.logging = get_root_logger(log_file=log_file, log_level='INFO')

    @rank_zero_only
    def log(self, metrics, step):
        if step % self.interval == 0:
            new_metrics = ''
            for k, v in metrics.items():
                k, v = str(k), str(v)
                new_metrics += f'{k} : {v} | '
            self.logging.info(new_metrics)

    @rank_zero_only
    def force_log(self, metrics, step):
        new_metrics = ''
        for k, v in metrics.items():
            new_metrics += f'{k} : {v} | '
        self.logging.info(new_metrics)
