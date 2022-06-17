from abc import abstractmethod

import mmcv
import numpy as np

from ..builder import DATASOURCES


@DATASOURCES.register_module()
class BaseDataSource(object):

    def __init__(self,
                 ann_file=None,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):

        self.ann_file = ann_file
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args
        self.file_client = None
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):

        if isinstance(self.ann_file, str):
            self.ann_file = [self.ann_file]

        data_infos = list()
        for ann_file in self.ann_file:
            with open(ann_file, 'r') as f:
                self.samples = f.readlines()

            for sample in self.samples:
                sample = sample.split()
                info = {'filename': sample[0]}
                data_infos.append(info)

        return data_infos

    def get_img(self, idx):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        filename = self.data_infos[idx]['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order)
        img = img.astype(np.uint8)
        return img

    def __len__(self):
        return len(self.data_infos)
