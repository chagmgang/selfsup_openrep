import os

from .base import BaseDataset
from .build import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class DINODataset(BaseDataset):

    def __init__(self,
                 global_pipelines,
                 local_pipelines,
                 ncrop_global_view,
                 ncrop_local_view,
                 img_dir,
                 img_suffix='.png',
                 data_root=None):

        self.global_pipelines = Compose(global_pipelines)
        self.local_pipelines = Compose(local_pipelines)
        self.ncrop_global_view = ncrop_global_view
        self.ncrop_local_view = ncrop_local_view
        self.img_dir = os.path.join(data_root,
                                    img_dir) if data_root else img_dir
        self.img_infos = self.load_images(self.img_dir, img_suffix)

    def __getitem__(self, i):

        global_views = [
            self.global_pipelines(self.img_infos[i])['img']
            for _ in range(self.ncrop_global_view)
        ]

        local_views = [
            self.local_pipelines(self.img_infos[i])['img']
            for _ in range(self.ncrop_local_view)
        ]

        return dict(global_views=global_views, local_views=local_views)
