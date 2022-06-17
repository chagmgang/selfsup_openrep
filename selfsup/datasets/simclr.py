from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class SimclrDataset(BaseDataset):

    def __init__(self, datasource, pipelines):
        super(SimclrDataset, self).__init__(
            datasource=datasource,
            pipelines=pipelines,
        )

    def __getitem__(self, idx):
        img = self.datasource.get_img(idx)
        img1 = self.pipelines(dict(img=img))['img']
        img2 = self.pipelines(dict(img=img))['img']
        return dict(img=[img1, img2])
