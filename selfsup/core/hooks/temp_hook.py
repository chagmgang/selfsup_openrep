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

        cur_iter = runner.iter
        iter_per_epoch = int(runner.max_iters / runner.max_epochs)
        warmup_iter = int(self.warmup_epoch * iter_per_epoch)

        cur_teacher_temp = self.get_linear(
            start_value=runner.model.module.start_teacher_temp,
            end_value=runner.model.module.end_teacher_temp,
            cur_iter=cur_iter,
            max_iter=warmup_iter,
        )

        cur_student_temp = self.get_linear(
            start_value=runner.model.module.start_student_temp,
            end_value=runner.model.module.end_student_temp,
            cur_iter=cur_iter,
            max_iter=warmup_iter,
        )

        runner.model.module.cur_teacher_temp = cur_teacher_temp
        runner.model.module.cur_student_temp = cur_student_temp

        if self.every_n_iters(runner, self.update_interval):

            runner.log_buffer.update({'cur_teacher_temp': cur_teacher_temp})
            runner.log_buffer.update({'cur_student_temp': cur_student_temp})


@HOOKS.register_module()
class iBOTTemperatureUpdateHook(DINOTemperatureUpdateHook):

    def before_train_iter(self, runner):
        for temp_name in [
                'start_teacher_cls_temp',
                'cur_teacher_cls_temp',
                'end_teacher_cls_temp',
                'start_teacher_patch_temp',
                'cur_teacher_patch_temp',
                'end_teacher_patch_temp',
                'start_student_cls_temp',
                'cur_student_cls_temp',
                'end_student_cls_temp',
                'start_student_patch_temp',
                'cur_student_patch_temp',
                'end_student_patch_temp',
        ]:
            assert hasattr(runner.model.module, temp_name), (
                f'The runner must have attribute {temp_name} in algorithms.'
            )  # noqa: E126

        cur_iter = runner.iter
        iter_per_epoch = int(runner.max_iters / runner.max_epochs)
        warmup_iter = int(self.warmup_epoch * iter_per_epoch)

        cur_teacher_patch_temp = self.get_linear(
            start_value=runner.model.module.start_teacher_patch_temp,
            end_value=runner.model.module.end_teacher_patch_temp,
            cur_iter=cur_iter,
            max_iter=warmup_iter,
        )

        cur_student_patch_temp = self.get_linear(
            start_value=runner.model.module.start_student_patch_temp,
            end_value=runner.model.module.end_student_patch_temp,
            cur_iter=cur_iter,
            max_iter=warmup_iter,
        )

        cur_teacher_cls_temp = self.get_linear(
            start_value=runner.model.module.start_teacher_cls_temp,
            end_value=runner.model.module.end_teacher_cls_temp,
            cur_iter=cur_iter,
            max_iter=warmup_iter,
        )

        cur_student_cls_temp = self.get_linear(
            start_value=runner.model.module.start_student_cls_temp,
            end_value=runner.model.module.end_student_cls_temp,
            cur_iter=cur_iter,
            max_iter=warmup_iter,
        )

        runner.model.module.cur_teacher_patch_temp = cur_teacher_patch_temp
        runner.model.module.cur_student_patch_temp = cur_student_patch_temp
        runner.model.module.cur_teacher_cls_temp = cur_teacher_cls_temp
        runner.model.module.cur_student_cls_temp = cur_student_cls_temp

        if self.every_n_iters(runner, self.update_interval):

            runner.log_buffer.update(
                {'cur_teacher_patch_temp': cur_teacher_patch_temp})
            runner.log_buffer.update(
                {'cur_student_patch_temp': cur_student_patch_temp})
            runner.log_buffer.update(
                {'cur_teacher_cls_temp': cur_teacher_cls_temp})
            runner.log_buffer.update(
                {'cur_student_cls_temp': cur_student_cls_temp})
