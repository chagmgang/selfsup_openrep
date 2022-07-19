from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class LinearWeightDecayUpdateHook(Hook):

    def __init__(self,
                 start_weight_decay,
                 end_weight_decay,
                 update_interval=1,
                 **kwargs):

        self.start_weight_decay = start_weight_decay
        self.end_weight_decay = end_weight_decay
        self.update_interval = update_interval

    def linear(self, start_value, end_value, cur_iter, max_iter):
        if cur_iter > max_iter:
            return end_value
        else:
            slope = (end_value - start_value) / max_iter
            x = cur_iter
            y = start_value
            return y + x * slope

    def before_train_iter(self, runner):

        weight_decay = self.linear(
            start_value=self.start_weight_decay,
            end_value=self.end_weight_decay,
            cur_iter=runner.iter,
            max_iter=runner.max_iters,
        )

        runner.optimizer.param_groups[0]['weight_decay'] = weight_decay

        if self.every_n_iters(runner, self.update_interval):
            weight_decay = runner.optimizer.param_groups[0]['weight_decay']
            runner.log_buffer.update({'weight_decay': weight_decay})
