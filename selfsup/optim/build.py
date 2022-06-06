import torch

import selfsup.optim.sched as sched
from .lars import LARS


def build_sched(optim, cfg):

    args = cfg.copy()
    obj_type = args.pop('type')

    func_type = getattr(sched, obj_type)
    scheduler = func_type(optimizer=optim, **args)
    return scheduler


def build_optim(params, cfg):

    args = cfg.copy()
    obj_type = args.pop('type')

    if obj_type == 'LARS':
        cls_type = LARS
    else:
        cls_type = getattr(torch.optim, obj_type)
    optimizer = cls_type(params=params, **args)

    return optimizer
