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
class MocoV3(BaseModel):

    def __init__(self,
                 backbone,
                 projection,
                 temperature=0.1,
                 start_momentum=0.996,
                 init_cfg=None):
        super(MocoV3, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.projection = build_projection(projection)
        self.prediction = build_projection(projection)

        self.momentum_backbone = build_backbone(backbone)
        self.momentum_projection = build_projection(projection)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.temperature = temperature
        self.base_momentum = start_momentum
        self.cur_momentum = start_momentum

        self.criterion = nn.CrossEntropyLoss()

        self.sync_models()

    def sync_models(self):

        def sync(src, dst):
            for param_src, param_dst in zip(src.parameters(),
                                            dst.parameters()):
                param_dst.data.copy_(param_src.data)
                param_dst.requires_grad = False

        sync(self.backbone, self.momentum_backbone)
        sync(self.projection, self.momentum_projection)

    def get_feature(self, img):
        x = self.backbone(img)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.projection(x)
        return self.prediction(x)

    def get_momentum_feature(self, img):
        x = self.momentum_backbone(img)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return self.momentum_projection(x)

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

    def extract_feat(self, img):
        pass

    @torch.no_grad()
    def momentum_update(self):

        def update(src, dst):
            for param_src, param_dst in zip(src.parameters(),
                                            dst.parameters()):
                param_dst.data = param_dst.data * self.cur_momentum + param_src.data * (
                    1. - self.cur_momentum)

        update(self.backbone, self.momentum_backbone)
        update(self.projection, self.momentum_projection)

    def forward_train(self, img, **kwargs):

        img1 = img[0]
        img2 = img[1]

        q1 = self.get_feature(img1)
        q2 = self.get_feature(img2)

        with torch.no_grad():
            k1 = self.get_momentum_feature(img1)
            k2 = self.get_momentum_feature(img2)

        loss = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        return dict(loss=loss)
