import torch
import torch.nn as nn
import torch.nn.functional as F

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
class DINO(BaseModel):

    def __init__(self,
                 backbone,
                 projection,
                 start_momentum=0.996,
                 center_momentum=0.9,
                 start_teacher_temp=0.04,
                 end_teacher_temp=0.07,
                 start_student_temp=0.1,
                 end_student_temp=0.1,
                 init_cfg=None):
        super(DINO, self).__init__(init_cfg)

        self.center_momentum = center_momentum
        self.center = torch.zeros(1, projection.last_dim).cuda()

        self.student_backbone = build_backbone(backbone)
        self.student_projection = build_projection(projection)

        self.backbone = build_backbone(backbone)  # teacher backbone
        self.projection = build_projection(projection)  # teacher projection

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        self.base_momentum = start_momentum
        self.cur_momentum = start_momentum

        self.start_teacher_temp = start_teacher_temp
        self.cur_teacher_temp = start_teacher_temp
        self.end_teacher_temp = end_teacher_temp

        self.start_student_temp = start_student_temp
        self.cur_student_temp = start_student_temp
        self.end_student_temp = end_student_temp

        self.sync_models()

    @torch.no_grad()
    def momentum_update(self):

        def update(src, dst):
            for param_src, param_dst in zip(src.parameters(),
                                            dst.parameters()):
                param_dst.data = param_dst.data * self.cur_momentum + param_src.data * (
                    1. - self.cur_momentum)

        update(self.student_backbone, self.backbone)
        update(self.student_projection, self.projection)

    def sync_models(self):

        def sync(src, dst):
            for param_src, param_dst in zip(src.parameters(),
                                            dst.parameters()):
                param_dst.data.copy_(param_src.data)
                param_dst.requires_grad = False

        sync(self.student_backbone, self.backbone)
        sync(self.student_projection, self.projection)

    def extract_feat(self, img):
        pass

    def get_teacher_feature(self, img):
        x = self.backbone(img)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.projection(x)
        return torch.nn.functional.normalize(x, dim=1)

    def get_student_feature(self, img):
        x = self.student_backbone(img)[-1]
        x = self.gap(x)
        x = self.flatten(x)
        x = self.student_projection(x)
        return torch.nn.functional.normalize(x, dim=1)

    def update_centering(self, teachers):
        all_teachers = [concat_all_gather(t) for t in teachers]
        batch = torch.cat(all_teachers)
        batch_mean = torch.mean(batch, dim=0, keepdim=True)
        self.center = self.center_momentum * self.center + (
            1 - self.center_momentum) * batch_mean

    def distillation(self, s, t):

        s = F.log_softmax(s / self.cur_student_temp, dim=-1)
        t = F.softmax((t - self.center) / self.cur_teacher_temp, dim=-1)
        loss = torch.sum(-t * s, dim=-1)
        loss = loss.mean()
        return loss

    def train_step(self, data, optimizer):
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        if isinstance(data['img']['global_views'], list):
            num_samples = len(data['img']['global_views'][0].data)
        else:
            num_samples = len(data['img']['global_views'].data)
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=num_samples)
        return outputs

    def forward_train(self, img, **kwargs):

        global_views = img['global_views']
        local_views = img['local_views']

        with torch.no_grad():
            global_rep = [
                self.get_teacher_feature(glb) for glb in global_views
            ]
            self.update_centering(global_rep)

        local_rep = [self.get_student_feature(loc) for loc in local_views]

        losses = 0
        losses_term = 0
        for glb in global_rep:
            for loc in local_rep:
                loss = self.distillation(loc, glb)
                losses += loss
                losses_term += 1

        return dict(loss=losses / losses_term)
