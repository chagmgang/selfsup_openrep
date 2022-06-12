from selfsup.utils import get_root_logger, print_log
from .base import BaseDataset
from .build import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class ListSimclrDataset(BaseDataset):

    def __init__(self, pipelines, txt_file):

        self.pipelines = Compose(pipelines)
        self.img_infos = self.load_images(txt_file)

    def load_images(self, txt_file):

        img_infos = list()
        f = open(txt_file)
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            img_info = dict(filename=line)
            img_infos.append(img_info)

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())

        return img_infos

    def __getitem__(self, i):
        data1 = self.pipelines(self.img_infos[i])
        data2 = self.pipelines(self.img_infos[i])
        return dict(img1=data1['img'], img2=data2['img'])

    def __len__(self):
        return len(self.img_infos)


@DATASETS.register_module()
class SimclrDataset(BaseDataset):

    def __getitem__(self, i):
        data1 = self.pipelines(self.img_infos[i])
        data2 = self.pipelines(self.img_infos[i])
        return dict(img1=data1['img'], img2=data2['img'])
