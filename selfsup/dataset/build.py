import torch
from torch.utils.data import ConcatDataset

from selfsup.utils import Registry, build_from_cfg

DATASETS = Registry('dataset')
PIPELINES = Registry('pipelines')


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_dataloader(dataset, cfg, world_size, global_rank, shuffle,
                     distributed):

    sampler = None

    if distributed:

        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=shuffle,
            drop_last=True,
        )

        shuffle = False

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.samples_per_gpu,
        num_workers=cfg.workers_per_gpu,
        drop_last=True,
        pin_memory=True,
        shuffle=shuffle,
        sampler=sampler,
    )

    return dataloader
