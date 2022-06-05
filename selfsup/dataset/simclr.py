import os

from selfsup.utils import get_root_logger, print_log
from .base import BaseDataset
from .build import DATASETS


@DATASETS.register_module()
class SimclrDataset(BaseDataset):

    def __init__(self, pipelines, img_dir, img_suffix='.png', data_root=None):
        super(SimclrDataset, self).__init__(
            pipelines, img_dir, img_suffix, data_root=None)

        self.pipelines = pipelines
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
        return self.img_infos[i]

    def __len__(self):
        return len(self.img_infos)
