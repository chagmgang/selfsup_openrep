from functools import partial

import torch
import torch.nn as nn

from ..builder import BACKBONES
from .utils import DropPath


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 bias=True,
                 drop=0.):
        super(Mlp, self).__init__()
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
                 image_size,
                 patch_size,
                 in_chans,
                 embed_dim,
                 norm_layer=None,
                 flatten=True):
        super(PatchEmbed, self).__init__()

        image_size = (image_size, image_size)
        patch_size = (patch_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0],
                          image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
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


class VisionTransformer(nn.Module):

    def __init__(
            self,
            image_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            global_pool='token',
            norm_layer=None,
            act_layer=None,
            fc_norm=None,
            class_token=True,
            embed_layer=PatchEmbed,
            drop_rate=0.,
            drop_path_rate=0.,
            block_fn=Block,
            mlp_ratio=4.,
            qkv_bias=True,
            init_values=None,
            attn_drop_rate=0.,
            out_index=(3, 5, 7, 11),
    ):
        super(VisionTransformer, self).__init__()

        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1 if class_token else 0
        self.grad_checkpointing = False
        self.out_index = out_index

        self.patch_embed = embed_layer(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if self.num_tokens > 0 else None
        self.pos_embed = self.build_2d_sincos_position_embedding()
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
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
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        self.drop_grad_patch_embed()

    def drop_grad_patch_embed(self):
        self.patch_embed.proj.weight.requires_grad = False
        self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0
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
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        pos_embed.requires_grad = False
        return pos_embed

    def forward(self, x):
        features = list()
        x = self.patch_embed(x)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x),
                          dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                features.append(x)
        x = self.norm(x)
        x = x[:, 0]
        x = x.unsqueeze(-1).unsqueeze(-1)
        features.append(x)
        return features


@BACKBONES.register_module()
class TinyViT(VisionTransformer):

    def __init__(self, image_size, patch_size):
        super(TinyViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            global_pool='token',
            embed_dim=192,
            depth=12,
            num_heads=3,
        )


@BACKBONES.register_module()
class SmallViT(VisionTransformer):

    def __init__(self, image_size, patch_size):
        super(SmallViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            global_pool='token',
            embed_dim=384,
            depth=12,
            num_heads=6,
        )


@BACKBONES.register_module()
class BaseViT(VisionTransformer):

    def __init__(self, image_size, patch_size):
        super(BaseViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            global_pool='token',
            embed_dim=768,
            depth=12,
            num_heads=12,
        )


@BACKBONES.register_module()
class LargeViT(VisionTransformer):

    def __init__(self, image_size, patch_size):
        super(LargeViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            global_pool='token',
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )


@BACKBONES.register_module()
class GiantViT(VisionTransformer):

    def __init__(self, image_size, patch_size):
        super(GiantViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            global_pool='token',
            embed_dim=1408,
            mlp_ratio=48 / 11,
            depth=40,
            num_heads=16,
        )


@BACKBONES.register_module()
class GiganticViT(VisionTransformer):

    def __init__(self, image_size, patch_size):
        super(GiganticViT, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=1664,
            global_pool='token',
            mlp_ratio=64 / 13,
            depth=48,
            num_heads=16,
        )
