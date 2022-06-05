import torch

from .build import DATASETS


@DATASETS.register_module()
class TestDataset(torch.utils.data.Dataset):

    def __init__(self):
        super(TestDataset, self).__init__()

        self.dataset = range(100)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return len(self.dataset)
