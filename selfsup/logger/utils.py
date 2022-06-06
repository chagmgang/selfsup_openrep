import torch.distributed as dist


def rank_zero_only(func):

    def decorated(*args, **kwargs):
        if dist.is_initialized():
            if dist.get_rank() == 0:
                func(*args, **kwargs)
        else:
            func(*args, **kwargs)

    return decorated
