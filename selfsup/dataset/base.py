import os

import torch

from selfsup.utils import get_root_logger, print_log
from .build import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, pipelines, img_dir, img_suffix='.png', data_root=None):
        super(BaseDataset, self).__init__()

        self.pipelines = Compose(pipelines)
        self.img_dir = os.path.join(data_root,
                                    img_dir) if data_root else img_dir
        self.img_infos = self.load_images(self.img_dir, img_suffix)

    def load_images(self, img_dir, img_suffix):

        img_infos = list()
        for root, dirs, files in os.walk(img_dir):
            for filename in files:
                if filename.endswith(img_suffix):
                    filename = os.path.join(root, filename)
                    img_info = dict(filename=filename)
                    img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())

        return img_infos

    def __getitem__(self, i):

        data1 = self.pipelines(self.img_infos[i])
        return dict(img1=data1['img'])

    def __len__(self):
        return len(self.img_infos)
