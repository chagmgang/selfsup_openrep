from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class DINOTemperatureUpdateHook(Hook):

    def __init__(self, warmup_epoch=30, update_interval=1, **kwargs):

        self.warmup_epoch = warmup_epoch
        self.update_interval = update_interval

    def get_linear(self, start_value, end_value, cur_iter, max_iter):

        if cur_iter > max_iter:
            return end_value
        else:
            slope = (end_value - start_value) / max_iter
            x = cur_iter
            y = start_value
            return y + x * slope

    def before_train_iter(self, runner):
        for temp_name in [
                'start_teacher_temp', 'cur_teacher_temp', 'end_teacher_temp',
                'start_student_temp', 'cur_student_temp', 'end_student_temp'
        ]:
            assert hasattr(runner.model.module, temp_name), (
                f'The runner must have attribute {temp_name} in algorithms.'
            )  # noqa: E126

        if self.every_n_iters(runner, self.update_interval):

            cur_iter = runner.iter
            max_iter = len(runner.data_loader) * self.warmup_epoch

            cur_teacher_temp = self.get_linear(
                start_value=runner.model.module.start_teacher_temp,
                end_value=runner.model.module.end_teacher_temp,
                cur_iter=cur_iter,
                max_iter=max_iter,
            )

            cur_student_temp = self.get_linear(
                start_value=runner.model.module.start_student_temp,
                end_value=runner.model.module.end_student_temp,
                cur_iter=cur_iter,
                max_iter=max_iter,
            )

            runner.model.module.cur_teacher_temp = cur_teacher_temp
            runner.model.module.cur_student_temp = cur_student_temp

            runner.log_buffer.update({'cur_teacher_temp': cur_teacher_temp})
            runner.log_buffer.update({'cur_student_temp': cur_student_temp})
