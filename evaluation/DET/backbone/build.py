from collections import OrderedDict

import torch
import torch.nn as nn
from mmrotate.models.builder import ROTATED_BACKBONES

from selfsup.models import backbone


@ROTATED_BACKBONES.register_module()
class SelfSupViT(nn.Module):

    def __init__(self,
                 model_name,
                 img_size,
                 patch_size,
                 weight=None,
                 unfreeze_patch=False,
                 freeze_all=False):
        super(SelfSupViT, self).__init__()

        model_module = getattr(backbone, model_name)
        self.model = model_module(
            img_size=img_size,
            patch_size=patch_size,
        )
        self.patch_size = patch_size

        if weight:
            model_state_dict = self.load_from(weight)
            model_state_dict = self.check_shape(model_state_dict)
            print(self.model.load_state_dict(model_state_dict, strict=False))

        self.fpn1 = nn.ConvTranspose2d(
            self.model.num_features,
            self.model.num_features,
            kernel_size=4,
            stride=4,
        )

        self.fpn2 = nn.ConvTranspose2d(
            self.model.num_features,
            self.model.num_features,
            kernel_size=2,
            stride=2,
        )

        self.fpn3 = nn.Identity()
        self.fpn4 = nn.Conv2d(
            self.model.num_features,
            self.model.num_features,
            kernel_size=2,
            stride=2)

    def check_shape(self, model_state_dict):
        state_dict = self.model.state_dict()
        new_state_dict = OrderedDict()
        for key in state_dict:
            original_shape = state_dict[key].shape
            copied_shape = model_state_dict[key].shape
            if original_shape == copied_shape:
                new_state_dict[key] = model_state_dict[key]
        return new_state_dict

    def load_from(self, weight):
        state_dict = torch.load(weight, map_location='cpu')['state_dict']
        new_state_dict = OrderedDict()

        for key in state_dict.keys():
            if key.startswith('backbone.'):
                value = state_dict[key]
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value
        return new_state_dict

    def forward(self, x):
        B, C, H, W = x.shape
        h_feat, w_feat = H // self.patch_size, W // self.patch_size
        features = self.model(x)[:4]
        features = [f[:, 1:] for f in features]
        features = [self.model.norm(f) for f in features]
        features = [f.permute(0, 2, 1) for f in features]
        features = [
            f.view(B, -1, h_feat, w_feat).contiguous() for f in features
        ]

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]

        features = [op(f) for f, op in zip(features, ops)]
        return tuple(features)


@ROTATED_BACKBONES.register_module()
class SelfSupBackbone(nn.Module):

    def __init__(self, model_name, pretrained=False, weight=None):
        super(SelfSupBackbone, self).__init__()

        model_module = getattr(backbone, model_name)
        self.model = model_module(pretrained=pretrained)

        if weight:
            model_state_dict = self.load_from(weight)
            print(self.model.load_state_dict(model_state_dict, strict=False))

    def load_from(self, weight):
        state_dict = torch.load(weight, map_location='cpu')
        model_state_dict = state_dict['state_dict']

        new_state_dict = OrderedDict()
        for key in model_state_dict.keys():
            value = model_state_dict[key]
            new_key = key.replace('backbone.', '')
            new_state_dict[new_key] = value

        return new_state_dict

    def forward(self, x):
        feature = self.model(x)
        return tuple(feature)
