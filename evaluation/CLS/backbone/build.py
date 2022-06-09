from collections import OrderedDict

import torch
import torch.nn as nn
from mmcls.models.builder import BACKBONES

from selfsup.models import backbone


@BACKBONES.register_module()
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
            new_key = key.replace('module.backbone.', '')
            new_state_dict[new_key] = value

        return new_state_dict
