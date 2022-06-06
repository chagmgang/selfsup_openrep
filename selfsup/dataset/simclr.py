from .base import BaseDataset
from .build import DATASETS


@DATASETS.register_module()
class SimclrDataset(BaseDataset):

    def __getitem__(self, i):
        data1 = self.pipelines(self.img_infos[i])
        data2 = self.pipelines(self.img_infos[i])
        return dict(img1=data1['img'], img2=data2['img'])
