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
class iBOT(BaseModel):

    def __init__(
        self,
        backbone,
        projection,
        mask_ratio=0.3,
        cls_center_momentum=0.9,
        patch_center_momentum=0.9,
        start_momentum=0.996,
        start_teacher_cls_temp=0.04,
        end_teacher_cls_temp=0.07,
        start_teacher_patch_temp=0.04,
        end_teacher_patch_temp=0.07,
        start_student_cls_temp=0.1,
        end_student_cls_temp=0.1,
        start_student_patch_temp=0.1,
        end_student_patch_temp=0.1,
        init_cfg=None,
    ):
        super(iBOT, self).__init__(init_cfg)

        self.mask_ratio = mask_ratio
        self.base_momentum = start_momentum
        self.cur_momentum = start_momentum

        self.start_teacher_cls_temp = start_teacher_cls_temp
        self.cur_teacher_cls_temp = start_teacher_cls_temp
        self.end_teacher_cls_temp = end_teacher_cls_temp

        self.start_teacher_patch_temp = start_teacher_patch_temp
        self.cur_teacher_patch_temp = start_teacher_patch_temp
        self.end_teacher_patch_temp = end_teacher_patch_temp

        self.start_student_cls_temp = start_student_cls_temp
        self.cur_student_cls_temp = start_student_cls_temp
        self.end_student_cls_temp = end_student_cls_temp

        self.start_student_patch_temp = start_student_patch_temp
        self.cur_student_patch_temp = start_student_patch_temp
        self.end_student_patch_temp = end_student_patch_temp

        self.cls_center_momentum = cls_center_momentum
        self.cls_center = torch.zeros(1, projection.out_dim).cuda()
        self.patch_center_momentum = patch_center_momentum
        self.patch_center = torch.zeros(1, projection.out_dim).cuda()

        self.backbone = build_backbone(backbone)
        self.cls_projection = build_projection(projection)
        self.patch_projection = build_projection(projection)

        self.student_backbone = build_backbone(backbone)
        self.student_cls_projection = build_projection(projection)
        self.student_patch_projection = build_projection(projection)

        self.mask_modeling = nn.Parameter(
            torch.zeros(1, self.backbone.num_features))

        self.sync_models()

    @torch.no_grad()
    def momentum_update(self):

        def update(src, dst):
            for param_src, param_dst in zip(src.parameters(),
                                            dst.parameters()):
                param_dst.data = param_dst.data * self.cur_momentum + param_src.data * (
                    1. - self.cur_momentum)

        update(self.student_backbone, self.backbone)
        update(self.student_cls_projection, self.cls_projection)
        update(self.student_patch_projection, self.patch_projection)

    def sync_models(self):

        def sync(src, dst):
            for param_src, param_dst in zip(src.parameters(),
                                            dst.parameters()):
                param_dst.data.copy_(param_src.data)
                param_dst.requires_grad = False

        sync(self.student_backbone, self.backbone)
        sync(self.student_cls_projection, self.cls_projection)
        sync(self.student_patch_projection, self.patch_projection)

    def update_centering(self, cls_features, patch_features):

        all_cls_features = [concat_all_gather(t) for t in cls_features]
        all_patch_features = [concat_all_gather(t) for t in patch_features]

        cls_batch = torch.cat(all_cls_features)
        patch_batch = torch.cat(all_patch_features)

        cls_batch_mean = torch.mean(cls_batch, dim=0, keepdim=True)
        patch_batch_mean = torch.mean(patch_batch, dim=0, keepdim=True)

        self.cls_center = self.cls_center_momentum * self.cls_center + (
            1 - self.cls_center_momentum) * cls_batch_mean
        self.patch_center = self.patch_center_momentum * self.patch_center + (
            1 - self.patch_center_momentum) * patch_batch_mean

    def extract_feat(self, img):
        pass

    def get_teacher_feature(self, img):
        feature = self.backbone(img)[-1]
        cls_feature, patch_feature = feature[:, 0], feature[:, 1:]
        cls_feature = self.cls_projection(cls_feature)
        patch_feature = self.patch_projection(patch_feature)

        batch_size, patches, dim = patch_feature.shape
        patch_feature = patch_feature.view(batch_size * patches, dim)
        return cls_feature, patch_feature

    def get_student_feature(self, img, mask=True):
        if mask:
            feature = self.student_backbone(
                img, mask_modeling=self.mask_modeling,
                ratio=self.mask_ratio)[-1]
        else:
            feature = self.student_backbone(img)[-1]
        cls_feature, patch_feature = feature[:, 0], feature[:, 1:]
        cls_feature = self.student_cls_projection(cls_feature)
        patch_feature = self.student_patch_projection(patch_feature)

        batch_size, patches, dim = patch_feature.shape
        patch_feature = patch_feature.view(batch_size * patches, dim)
        return cls_feature, patch_feature

    def distillation(self, s, t, s_temp, t_temp, center):
        s = F.log_softmax(s / s_temp, dim=-1)
        t = F.softmax((t - center) / t_temp, dim=-1)
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
            teacher_cls_features, teacher_patch_features = list(), list()
            for glb in global_views:
                cls_feature, patch_feature = self.get_teacher_feature(glb)
                teacher_cls_features.append(cls_feature)
                teacher_patch_features.append(patch_feature)
            self.update_centering(teacher_cls_features, teacher_patch_features)

        student_cls_features, student_patch_features = list(), list()
        for glb in global_views:
            cls_feature, patch_feature = self.get_student_feature(
                glb, mask=True)
            student_cls_features.append(cls_feature)
            student_patch_features.append(patch_feature)

        student_local_cls_features = list()
        for loc in local_views:
            cls_feature, _ = self.get_student_feature(loc)
            student_local_cls_features.append(cls_feature)

        losses = 0
        losses_term = 0

        # loss between global views

        for i, (cls_t, patch_t) in enumerate(
                zip(teacher_cls_features, teacher_patch_features)):
            for j, (cls_s, patch_s) in enumerate(
                    zip(student_cls_features, student_patch_features)):
                if i == j:  # patch loss between global views
                    loss = self.distillation(
                        s=patch_s,
                        t=patch_t,
                        s_temp=self.cur_student_patch_temp,
                        t_temp=self.cur_teacher_patch_temp,
                        center=self.patch_center,
                    )
                    losses += loss
                    losses_term += 1
                else:  # cls loss between global views
                    loss = self.distillation(
                        s=cls_s,
                        t=cls_t,
                        s_temp=self.cur_student_cls_temp,
                        t_temp=self.cur_teacher_cls_temp,
                        center=self.cls_center,
                    )
                    losses += loss
                    losses_term += 1

        # loss between teacher global view and student local view
        for i, cls_t in enumerate(teacher_cls_features):
            for j, cls_s in enumerate(student_local_cls_features):
                loss = self.distillation(
                    s=cls_s,
                    t=cls_t,
                    s_temp=self.cur_student_patch_temp,
                    t_temp=self.cur_teacher_patch_temp,
                    center=self.cls_center,
                )
                losses += loss
                losses_term += 1

        return dict(loss=losses / losses_term)
