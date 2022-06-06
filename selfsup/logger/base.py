from selfsup.logger import LOGGERS
from .utils import rank_zero_only


@LOGGERS.register_module()
class BaseLogger(object):

    def __init__(self, **kwargs):
        super(BaseLogger, self).__init__()

    @rank_zero_only
    def before_run(self):
        pass

    @rank_zero_only
    def log(self, metrics, step):
        pass

    @rank_zero_only
    def force_log(self, metrics, step):
        pass

    @rank_zero_only
    def after_run(self):
        pass
