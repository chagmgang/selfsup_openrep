from .base import BaseDataset
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class SimclrDataset(BaseDataset):

    def __init__(self, pipelines, img_dir, img_suffix='.png', data_root=None):
        super(SimclrDataset, self).__init__(
            pipelines=pipelines,
            img_dir=img_dir,
            img_suffix=img_suffix,
            data_root=data_root,
        )

    def __getitem__(self, idx):
        data1 = self.pipelines(self.img_infos[idx])
        data2 = self.pipelines(self.img_infos[idx])
        img1 = data1['img']
        img2 = data2['img']
        return dict(img=[img1, img2])


@DATASETS.register_module()
class ListSimclrDataset(SimclrDataset):

    def __init__(self, pipelines, txt_file):

        self.pipelines = Compose(pipelines)
        self.txt_file = txt_file

        self.img_infos = self.load_images(txt_file)

    def load_images(self, txt_file):

        f = open(txt_file, 'r')
        img_infos = list()
        while True:
            line = f.readline()
            if not line:
                break
            line = line.replace('\n', '')
            img_info = dict(filename=line)
            img_infos.append(img_info)

        f.close()

        return img_infos
