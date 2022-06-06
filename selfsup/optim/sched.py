import math

from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(optimizer,
                       num_warmup_steps,
                       num_training_steps,
                       num_cycles=0.5,
                       last_epoch=-1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0,
            0.5 *  # noqa : W504
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
