from typing import Callable, List, Optional, Type, Union

import torch.nn as nn
from torch import Tensor

from selfsup.models import BACKBONE
from selfsup.utils._internally_replaced_utils import load_state_dict_from_url

model_urls = {
    'resnet18':
    'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34':
    'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50':
    'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101':
    'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152':
    'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d':
    'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d':
    'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2':
    'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2':
    'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        use_classifier: bool = False,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2])

        if use_classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight,
                                      0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight,
                                      0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        features = list()
        for i in range(4):
            layer = getattr(self, f'layer{i+1}')
            x = layer(x)
            features.append(x)

        return features

    def pretrained(self, model_name):
        state_dict = load_state_dict_from_url(
            model_urls[model_name], progress=True)
        unfined_key, missing_key = self.load_state_dict(
            state_dict, strict=False)
        print(f'unfined_key : {unfined_key}')
        print(f'missing_key : {missing_key}')


@BACKBONE.register_module()
class ResNet18(ResNet):

    def __init__(self, **kwargs):
        super(ResNet18, self).__init__(block=BasicBlock, layers=[2, 2, 2, 2])

        if kwargs['pretrained']:
            self.pretrained('resnet18')


@BACKBONE.register_module()
class ResNet34(ResNet):

    def __init__(self, **kwargs):
        super(ResNet34, self).__init__(block=BasicBlock, layers=[3, 4, 6, 3])

        if kwargs['pretrained']:
            self.pretrained('resnet34')


@BACKBONE.register_module()
class ResNet50(ResNet):

    def __init__(self, **kwargs):
        super(ResNet50, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3])

        if kwargs['pretrained']:
            self.pretrained('resnet50')


@BACKBONE.register_module()
class ResNet101(ResNet):

    def __init__(self, **kwargs):
        super(ResNet101, self).__init__(block=Bottleneck, layers=[3, 4, 23, 3])

        if kwargs['pretrained']:
            self.pretrained('resnet101')


@BACKBONE.register_module()
class ResNet152(ResNet):

    def __init__(self, **kwargs):
        super(ResNet152, self).__init__(block=Bottleneck, layers=[3, 8, 36, 3])

        if kwargs['pretrained']:
            self.pretrained('resnet152')


@BACKBONE.register_module()
class ResNext50_32x4d(ResNet):

    def __init__(self, **kwargs):

        model_kwargs = dict()
        model_kwargs['groups'] = 32
        model_kwargs['width_per_group'] = 4

        super(ResNext50_32x4d, self).__init__(
            block=Bottleneck, layers=[3, 4, 6, 3], **model_kwargs)

        if kwargs['pretrained']:
            self.pretrained('resnext50_32x4d')


@BACKBONE.register_module()
class ResNext101_32x8d(ResNet):

    def __init__(self, **kwargs):

        model_kwargs = dict()
        model_kwargs['groups'] = 32
        model_kwargs['width_per_group'] = 8

        super(ResNext101_32x8d, self).__init__(
            block=Bottleneck, layers=[3, 4, 23, 3], **model_kwargs)

        if kwargs['pretrained']:
            self.pretrained('resnext101_32x8d')


@BACKBONE.register_module()
class WideResNet50_2(ResNet):

    def __init__(self, **kwargs):

        model_kwargs = dict()
        model_kwargs['width_per_group'] = 64 * 2

        super(WideResNet50_2, self).__init__(
            block=Bottleneck, layers=[3, 4, 6, 3], **model_kwargs)

        if kwargs['pretrained']:
            self.pretrained('wide_resnet50_2')


@BACKBONE.register_module()
class WideResNet101_2(ResNet):

    def __init__(self, **kwargs):

        model_kwargs = dict()
        model_kwargs['width_per_group'] = 64 * 2

        super(WideResNet101_2, self).__init__(
            block=Bottleneck, layers=[3, 4, 23, 3], **model_kwargs)

        if kwargs['pretrained']:
            self.pretrained('wide_resnet101_2')
