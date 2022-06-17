from abc import ABCMeta

from torch.utils.data import Dataset

from .builder import DATASETS, build_datasource
from .pipelines import Compose


@DATASETS.register_module()
class BaseDataset(Dataset):

    def __init__(self, datasource, pipelines):

        self.datasource = build_datasource(datasource)
        self.pipelines = Compose(pipelines)

    def __len__(self):
        return len(self.datasource)

    def __getitem__(self, idx):
        img = self.datasource.get_img(idx)
        img = self.pipelines(dict(img=img))['img']
        return dict(img=img)


@DATASETS.register_module()
class TestDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, prefetch=False):
        super(TestDataset, self).__init__()

        self.data_infos = self.load_annotations()

    def load_annotations(self):
        return [i for i in range(100)]

    def __getitem__(self, idx):
        return dict(img=self.data_infos[idx])

    def __len__(self):
        return len(self.data_infos)
