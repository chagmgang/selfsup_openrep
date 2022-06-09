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
            torchvision.datasets.STL10(
                root='data',
                split='train',
                download=True,
                transform=[],
            )

        if world_size > 1:
            dist.barrier()
            assert self._check_integrity(), \
                'Shared storage seems unavailable. '
