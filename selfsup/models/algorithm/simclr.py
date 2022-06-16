import torch
import torch.distributed as dist
import torch.nn as nn

from ..builder import ALGORITHMS, build_backbone, build_projection
from .base import BaseModel


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [
            torch.zeros_like(input) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


@ALGORITHMS.register_module()
class Simclr(BaseModel):

    def __init__(self, backbone, projection, temperature=0.1, init_cfg=None):
        super(Simclr, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.projection = build_projection(projection)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.global_step = 0
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def get_projection(self, x):
        x = self.extract_feat(x)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        return self.projection(x)

    def contrastive_loss(self, pos, neg):
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).to(pos.device)
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses

    @staticmethod
    def _create_buffer(N):
        """Compute the mask and the index of positive samples.

        Args:
            N (int): batch size.
        """
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        img = torch.stack([img[0], img[1]], 1)
        img = img.reshape(
            (img.size(0) * 2, img.size(2), img.size(3), img.size(4)))
        z = self.get_projection(img)
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.contrastive_loss(positive, negative)
        return losses
