import torch
import torch.distributed as dist
import torch.nn as nn

from selfsup.models import ALGORITHM, build_backbone, build_projection


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


@ALGORITHM.register_module()
class Simclr(nn.Module):

    def __init__(self, backbone, projection, tau=0.1):
        super(Simclr, self).__init__()

        self.backbone = build_backbone(backbone)
        self.projection = build_projection(projection)
        self.tau = tau

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.criterion = nn.CrossEntropyLoss()

    def get_projected_representation(self, x):

        x = self.backbone(x)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.projection(x)

        return x

    def contrastive_loss(self, q, k):
        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)

        if dist.is_initialized():
            k = concat_all_gather(k)

        logits = torch.einsum('nc,mc->nm', [q, k])
        N = logits.shape[0]

        labels = torch.arange(N, dtype=torch.long)
        if dist.is_initialized():
            labels += N * torch.distributed.get_rank()

        labels = labels.cuda()
        return self.criterion(logits, labels) * 2 * self.tau

    def forward_train(self, batch):

        x1 = batch['img1']
        x2 = batch['img2']

        q1 = self.get_projected_representation(x1)
        q2 = self.get_projected_representation(x2)

        return self.contrastive_loss(q1, q2) + self.contrastive_loss(q2, q1)

    def forward(self, x, train=True):
        if train:
            return self.forward_train(x)
        else:
            return self.forward_test(x)
