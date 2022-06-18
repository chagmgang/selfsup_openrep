from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class MomentumUpdateHook(Hook):

    def __init__(self, end_momentum=1., update_interval=1, **kwargs):

        self.end_momentum = end_momentum
        self.update_interval = update_interval

    def before_train_iter(self, runner):
        assert hasattr(runner.model.module, 'base_momentum'), (
            "The runner must have attribute 'base_momentum' in algorithms."
        )  # noqa: E126
        assert hasattr(runner.model.module, 'cur_momentum'), (
            "The runner must have attribute 'cur_momentum' in algorithms."
        )  # noqa: E126
        if self.every_n_iters(runner, self.update_interval):
            cur_iter = runner.iter
            max_iter = runner.max_iters
            base_m = runner.model.module.base_momentum
            m = self.end_momentum - (self.end_momentum - base_m) * (
                cos(pi * cur_iter / float(max_iter)) + 1) / 2
            runner.model.module.cur_momentum = m
            runner.log_buffer.update({'current_momentum': m})
            runner.log_buffer.update({'base_momentum': base_m})

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.update_interval):
            if is_module_wrapper(runner.model):
                runner.model.module.momentum_update()
            else:
                runner.model.momentum_update()
