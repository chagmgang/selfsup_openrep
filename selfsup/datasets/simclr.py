import torchvision

from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class SimclrDataset(BaseDataset):

    def __init__(self, pipelines, img_dir, img_suffix='.png', data_root=None):
        super(SimclrDataset, self).__init__(
            pipelines=pipelines,
            img_dir=img_dir,
            img_suffix=img_suffix,
            data_root=data_root,
        )

        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        data1 = self.pipelines(self.img_infos[idx])
        data2 = self.pipelines(self.img_infos[idx])
        img1 = data1['img']
        img2 = data2['img']
        return dict(img=[img1, img2])
