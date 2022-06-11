import math

import torch.nn as nn
from torch.nn import functional as F

from selfsup.models import ALGORITHM, build_backbone, build_projection


@ALGORITHM.register_module()
class BYOL(nn.Module):

    def __init__(self, backbone, projection, init_tau=0.996):
        super(BYOL, self).__init__()

        self.backbone = build_backbone(backbone)
        self.projection = build_projection(projection)
        self.prediction = build_projection(projection)

        self.target_backbone = build_backbone(backbone)
        self.target_projection = build_projection(projection)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.init_tau = init_tau

        self.sync_params(self.backbone, self.target_backbone)
        self.sync_params(self.projection, self.target_projection)

    def sync_params(self, source, target):

        for param_source, param_target in zip(source.parameters(),
                                              target.parameters()):
            param_target.data.copy_(param_source.data)
            param_target.requires_grad = False

    def target_representation(self, x):

        x = self.target_backbone(x)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.target_projection(x)

        return x

    def online_representation(self, x):

        x = self.backbone(x)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.projection(x)
        x = self.prediction(x)

        return x

    def update_tau(self, current_step, max_step):
        tau = 1 - (1 - self.init_tau) * (
            math.cos(math.pi * current_step / max_step) + 1) / 2
        return tau

    def update_weight(self, online_net, target_net, tau):

        for (online_name, online_p), (_, target_p) in zip(
                online_net.named_parameters(),
                target_net.named_parameters(),
        ):
            target_p.data = tau * target_p.data + (1 - tau) * online_p.data

    def mse(self, q, k):

        loss = -2 * F.cosine_similarity(q, k).mean()

        return loss

    def forward_train(self, batch, **kwargs):

        current_tau = self.update_tau(
            current_step=kwargs['global_step'],
            max_step=kwargs['max_step'],
        )

        self.update_weight(self.backbone, self.target_backbone, current_tau)
        self.update_weight(self.projection, self.target_projection,
                           current_tau)

        x1 = batch['img1']
        x2 = batch['img2']

        o1 = self.online_representation(x1)
        o2 = self.online_representation(x2)

        t1 = self.target_representation(x1)
        t2 = self.target_representation(x2)

        return self.mse(o1, t2) + self.mse(o2, t1)

    def forward(self, x, train=True, **kwargs):
        if train:
            return self.forward_train(x, **kwargs)
        else:
            return self.forward_test(x)
