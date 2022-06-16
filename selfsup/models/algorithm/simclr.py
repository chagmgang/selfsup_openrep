import os

import psutil
import torch
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_projection
from .base import BaseModel


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@ALGORITHMS.register_module()
class Simclr(BaseModel):

    def __init__(self, backbone, projection, temperature=0.1, init_cfg=None):
        super(Simclr, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.projection = build_projection(projection)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def get_projection(self, x):
        x = self.extract_feat(x)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return self.projection(x)

    def contrastive_loss(self, q, k):
        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)
        k = concat_all_gather(k)

        logits = torch.einsum('nc,mc->nm', [q, k])
        N = logits.shape[0]
        labels = (
            torch.arange(N, dtype=torch.long) +  # noqa: W504
            N * torch.distributed.get_rank()).cuda()
        return self.criterion(logits, labels) * 2 * self.temperature

    def forward_train(self, img, **kwargs):

        img1 = img[0]
        img2 = img[1]

        q1 = self.get_projection(img1)
        q2 = self.get_projection(img2)

        loss = self.contrastive_loss(q1, q2) + self.contrastive_loss(q2, q1)
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        print(memory_usage)
        return dict(loss=loss)
