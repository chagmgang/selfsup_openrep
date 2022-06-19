from torch.utils.data import Dataset

from .builder import DATASETS, build_datasource
from .pipelines import Compose


@DATASETS.register_module()
class DinoDataset(Dataset):

    def __init__(self,
                 datasource,
                 global_pipelines,
                 local_pipelines,
                 global_ncrop=2,
                 local_ncrop=6):

        self.datasource = build_datasource(datasource)

        self.global_pipelines = Compose(global_pipelines)
        self.local_pipelines = Compose(local_pipelines)

        self.global_ncrop = global_ncrop
        self.local_ncrop = local_ncrop

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, idx):
        img = self.datasource.get_img(idx)
        global_views = [
            self.global_pipelines(dict(img=img))['img']
            for _ in range(self.global_ncrop)
        ]
        local_views = [
            self.local_pipelines(dict(img=img))['img']
            for _ in range(self.local_ncrop)
        ]
        return dict(
            img=dict(global_views=global_views, local_views=local_views))
