import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from selfsup import Config
from selfsup.dataset import build_dataloader, build_dataset


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()

    return port


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--num-machines', default=1, type=int)
    parser.add_argument('--gpus-per-machines', default=1, type=int)
    parser.add_argument('--machine-rank', default=0, type=int)
    parser.add_argument('--load-from', type=str)
    parser.add_argument('--dist-url', type=str)
    args = parser.parse_args()

    return args


def launch(func, num_machines, gpus_per_machine, machine_rank, dist_url, args):

    world_size = num_machines * gpus_per_machine
    if world_size == 1:
        func(
            config_file=args.config_file,
            global_rank=0,
            world_size=world_size,
            distributed=False,
            load_from=args.load_from,
        )
    else:
        free_port = _find_free_port()
        dist_url = f'{dist_url}:{free_port}' if dist_url else f'tcp://127.0.0.1:{free_port}'
        mp.spawn(
            _distributed_worker,
            nprocs=gpus_per_machine,
            args=(func, world_size, gpus_per_machine, machine_rank, dist_url,
                  args),
            daemon=False,
        )


def _distributed_worker(local_rank, func, world_size, gpus_per_machine,
                        machine_rank, dist_url, args):

    global_rank = machine_rank * gpus_per_machine + local_rank
    dist.init_process_group(
        backend='NCCL',
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
    )

    torch.cuda.set_device(local_rank)
    func(
        config_file=args.config_file,
        global_rank=global_rank,
        world_size=world_size,
        distributed=True,
        load_from=args.load_from,
    )


def train(config_file, global_rank, world_size, distributed, load_from):

    cfg = Config.fromfile(config_file)

    import numpy as np
    from PIL import Image
    dataset = build_dataset(cfg.data.train)
    for i in range(10):
        data = dataset[i]
        img1 = data['img1']
        img1 = np.array(img1, dtype=np.uint8)
        img1 = Image.fromarray(img1).convert('RGB')
        img1.save(f'{global_rank}_{i}.png')

    dataloader = build_dataloader(
        dataset=build_dataset(cfg.data.train),
        cfg=cfg.data,
        world_size=world_size,
        global_rank=global_rank,
        shuffle=True,
        distributed=distributed,
    )

    for epoch in range(2):

        if isinstance(dataloader.sampler,
                      torch.utils.data.distributed.DistributedSampler):

            dataloader.sampler.set_epoch(epoch)

        for data in dataloader:

            print(epoch, data, global_rank)


if __name__ == '__main__':

    args = parse_args()
    launch(
        train,
        num_machines=args.num_machines,
        gpus_per_machine=args.gpus_per_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=args,
    )
