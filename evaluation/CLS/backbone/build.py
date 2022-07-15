from collections import OrderedDict

import torch
import torch.nn as nn
from mmcls.models.builder import BACKBONES

from selfsup.models import backbone


@BACKBONES.register_module()
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

        if weight:
            model_state_dict = self.load_from(weight)
            print(self.model.load_state_dict(model_state_dict, strict=False))

        if unfreeze_patch:
            self.unfreeze_patch_embed()

        if freeze_all:
            self.freeze_all()

    def freeze_all(self):

        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_patch_embed(self):
        self.model.patch_embed.proj.weight.requires_grad = True
        self.model.patch_embed.proj.bias.requires_grad = True

    def load_from(self, weight):
        state_dict = torch.load(weight, map_location='cpu')
        model_state_dict = state_dict['state_dict']

        new_state_dict = OrderedDict()
        for key in model_state_dict.keys():
            if key.startswith('backbone.'):
                value = model_state_dict[key]
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = value

        return new_state_dict

    def forward(self, x):
        feature = self.model(x)
        return tuple(feature)


@BACKBONES.register_module()
class SelfSupBackbone(nn.Module):

    def __init__(self,
                 model_name,
                 pretrained=False,
                 weight=None,
                 freeze_all=False):
        super(SelfSupBackbone, self).__init__()

        model_module = getattr(backbone, model_name)
        self.model = model_module(pretrained=pretrained)

        if weight:
            model_state_dict = self.load_from(weight)
            print(self.model.load_state_dict(model_state_dict, strict=False))

        if freeze_all:
            self.freeze_all()

    def freeze_all(self):

        for param in self.parameters():
            param.requires_grad = False

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
