import torch
import torch.nn as nn
from mmcv.runner import get_dist_info

from ..builder import ALGORITHMS, build_backbone, build_projection
from .base import BaseModel


@ALGORITHMS.register_module()
class Simclr(BaseModel):

    def __init__(self, backbone, projection, init_cfg=None):
        super(Simclr, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.projection = build_projection(projection)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.global_step = 0

    def extract_feat(self, img):
        x = self.backbone(img)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x

    def forward_train(self, img, **kwargs):
        rank, world_size = get_dist_info()
        import torchvision
        imgs = list()
        for i in img[0]:
            imgs.append(i)
        for i in img[1]:
            imgs.append(i)
        torchvision.utils.save_image(imgs, f'{self.global_step}_{rank}.png')
        img = torch.rand([5, 3, 224, 224]).cuda()
        x = self.extract_feat(img)
        y = torch.rand([5, 2048]).cuda()
        loss = (y - x) * (y - x)
        self.global_step += 1
        return dict(loss=torch.mean(loss))
