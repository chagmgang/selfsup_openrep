import numpy as np
import torch.distributed as dist
import torchvision
from mmcls.datasets import BaseDataset
from mmcls.datasets.builder import DATASETS
from mmcv.runner import get_dist_info


@DATASETS.register_module()
class TrainTorchvisionSTL10(BaseDataset):

    def load_annotations(self):

        rank, world_size = get_dist_info()
        if rank == 0:
            dataset = torchvision.datasets.STL10(
                root='data',
                split='train',
                download=True,
            )

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. '

        data_infos = list()
        for i in range(len(dataset)):
            img, target = dataset[i]
            img = np.array(img)
            gt_label = np.array(target, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)

        return data_infos


@DATASETS.register_module()
class TestTorchvisionSTL10(BaseDataset):

    def load_annotations(self):

        rank, world_size = get_dist_info()
        if rank == 0:
            dataset = torchvision.datasets.STL10(
                root='data',
                split='test',
                download=True,
            )

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. '

        data_infos = list()
        for i in range(len(dataset)):
            img, target = dataset[i]
            img = np.array(img)
            gt_label = np.array(target, dtype=np.int64)
            info = {'img': img, 'gt_label': gt_label}
            data_infos.append(info)

        return data_infos
