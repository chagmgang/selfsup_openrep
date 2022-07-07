from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES


def drop_block_2d(x,
                  drop_prob: float = 0.1,
                  block_size: int = 7,
                  gamma_scale: float = 1.0,
                  with_noise: bool = False,
                  inplace: bool = False,
                  batchwise: bool = False):
    """DropBlock.

    See https://arxiv.org/pdf/1810.12890.pdf DropBlock with an experimental
    gaussian noise option. This layer has been tested on a few training runs
    with success, but needs further validation and possibly optimization for
    lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size**2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(
        torch.arange(W).to(x.device),
        torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W),
                                   dtype=x.dtype,
                                   device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(
        dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn(
            (1, C, H, W), dtype=x.dtype,
            device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (
            block_mask.numel() /  # noqa: W504
            block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


def drop_block_fast_2d(x: torch.Tensor,
                       drop_prob: float = 0.1,
                       block_size: int = 7,
                       gamma_scale: float = 1.0,
                       with_noise: bool = False,
                       inplace: bool = False):
    """DropBlock.

    See https://arxiv.org/pdf/1810.12890.pdf DropBlock with an experimental
    gaussian noise option. Simplied from above without concern for valid block
    mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size**2 / (
        (W - block_size + 1) * (H - block_size + 1))

    block_mask = torch.empty_like(x).bernoulli_(gamma)
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype),
        kernel_size=clipped_block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.empty_like(x).normal_()
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (
            block_mask.numel() /  # noqa: W504
            block_mask.to(dtype=torch.float32).sum().add(1e-6)).to(
                dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x


class DropBlock2d(nn.Module):
    """DropBlock.

    See https://arxiv.org/pdf/1810.12890.pdf
    """

    def __init__(self,
                 drop_prob: float = 0.1,
                 block_size: int = 7,
                 gamma_scale: float = 1.0,
                 with_noise: bool = False,
                 inplace: bool = False,
                 batchwise: bool = False,
                 fast: bool = True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(x, self.drop_prob, self.block_size,
                                      self.gamma_scale, self.with_noise,
                                      self.inplace)
        else:
            return drop_block_2d(x, self.drop_prob, self.block_size,
                                 self.gamma_scale, self.with_noise,
                                 self.inplace, self.batchwise)


def drop_path(x,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 bias=True,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans,
                 embed_dim,
                 norm_layer=None,
                 flatten=True):
        super(PatchEmbed, self).__init__()

        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[
            0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[
            1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):

    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 init_values=None,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)
        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class ResPostBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 init_values=None,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.init_values = init_values

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop)
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.init_weights()

    def init_weights(self):
        # NOTE this init overrides that base model init with specific changes for the block type
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def forward(self, x):
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class ParallelBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 num_parallel=2,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 init_values=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(
                nn.Sequential(
                    OrderedDict([('norm', norm_layer(dim)),
                                 ('attn',
                                  Attention(
                                      dim,
                                      num_heads=num_heads,
                                      qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,
                                      proj_drop=drop)),
                                 ('ls',
                                  LayerScale(dim, init_values=init_values)
                                  if init_values else nn.Identity()),
                                 ('drop_path', DropPath(drop_path)
                                  if drop_path > 0. else nn.Identity())])))
            self.ffns.append(
                nn.Sequential(
                    OrderedDict([('norm', norm_layer(dim)),
                                 ('mlp',
                                  Mlp(dim,
                                      hidden_features=int(dim * mlp_ratio),
                                      act_layer=act_layer,
                                      drop=drop)),
                                 ('ls',
                                  LayerScale(dim, init_values=init_values)
                                  if init_values else nn.Identity()),
                                 ('drop_path', DropPath(drop_path)
                                  if drop_path > 0. else nn.Identity())])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class MocoV3ViT(nn.Module):

    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans=3,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 init_values=None,
                 class_token=True,
                 fc_norm=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 block_fn=Block,
                 out_index=(3, 5, 7, 11)):
        super(MocoV3ViT, self).__init__()

        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.out_index = out_index

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1 if class_token else 0

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.build_2d_sincos_position_embedding()
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer) for i in range(depth)
        ])

        self.freeze_patch_embed()

    def freeze_patch_embed(self):
        self.patch_embed.proj.weight.requires_grad = False
        self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([
            torch.sin(out_w),
            torch.cos(out_w),
            torch.sin(out_h),
            torch.cos(out_h)
        ],
                            dim=1)[None, :, :]  # noqa: E126

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def forward(self, x):
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x),
                          dim=1)
        x = self.pos_drop(x + self.pos_embed)
        features = list()
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                features.append(x)

        last_feature = x[:, 0]
        last_feature = last_feature.unsqueeze(-1).unsqueeze(-1)
        features.append(last_feature)
        return features


@BACKBONES.register_module()
class TinyMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(TinyMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=12,
            num_heads=3,
        )


@BACKBONES.register_module()
class SmallMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(SmallMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
        )


@BACKBONES.register_module()
class BaseMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(BaseMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )


@BACKBONES.register_module()
class LargeMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(LargeMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )


@BACKBONES.register_module()
class HugeMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(LargeMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1280,
            depth=32,
            num_heads=16,
        )


@BACKBONES.register_module()
class GiantMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(GiantMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1408,
            mlp_ratio=48 / 11,
            depth=40,
            num_heads=16,
        )


@BACKBONES.register_module()
class GiganticMocoV3ViT(MocoV3ViT):

    def __init__(self, img_size, patch_size):
        super(GiganticMocoV3ViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1664,
            mlp_ratio=64 / 13,
            depth=48,
            num_heads=16,
        )
