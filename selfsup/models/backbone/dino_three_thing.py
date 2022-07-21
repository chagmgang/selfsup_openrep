import math
from functools import partial

import torch
import torch.nn as nn

from ..builder import BACKBONES


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HMLPStem(nn.Module):

    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=nn.SyncBatchNorm):
        super(HMLPStem, self).__init__()

        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=in_chans,
                out_channels=embed_dim // 4,
                kernel_size=patch_size // 4,
                stride=patch_size // 4,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=embed_dim // 4,
                out_channels=embed_dim // 4,
                kernel_size=patch_size // 8,
                stride=patch_size // 8,
            ),
            # norm_layer(embed_dim // 4),
            nn.GELU(),
            nn.Conv2d(
                in_channels=embed_dim // 4,
                out_channels=embed_dim,
                kernel_size=patch_size // 8,
                stride=patch_size // 8,
            ),
            nn.GELU(),
            # norm_layer(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.ModuleList):

    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                 proj_drop):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ParallelBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio,
                 qkv_bias,
                 qk_scale,
                 drop,
                 attn_drop,
                 drop_path,
                 norm_layer,
                 act_layer,
                 parallel,
                 init_values=1e-4):
        super(ParallelBlock, self).__init__()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norms_1 = nn.ModuleList(
            [norm_layer(dim) for _ in range(parallel)])
        self.gammas_1 = nn.ParameterList([
            nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            for _ in range(parallel)
        ])
        self.attns = nn.ModuleList([
            Attention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop) for _ in range(parallel)
        ])

        self.norms_2 = nn.ModuleList(
            [norm_layer(dim) for _ in range(parallel)])
        self.gammas_2 = nn.ParameterList([
            nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            for _ in range(parallel)
        ])
        self.mlps = nn.ModuleList([
            Mlp(in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=drop) for _ in range(parallel)
        ])

    def forward(self, x):

        # MHSA Parallel
        mhsa_features = list()
        for norm, attn, gamma in zip(self.norms_1, self.attns, self.gammas_1):
            f = self.drop_path(gamma * attn(norm(x)))
            mhsa_features.append(f)

        for f in mhsa_features:
            x += f

        # ffn parallel
        ffn_features = list()
        for norm, mlp, gamma in zip(self.norms_2, self.mlps, self.gammas_2):
            f = self.drop_path(gamma * mlp(norm(x)))
            ffn_features.append(f)

        for f in ffn_features:
            x += f

        return x


class VisionTransformer(nn.Module):

    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 parallel=2,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 out_index=(3, 5, 7, 11),
                 **kwargs):
        super(VisionTransformer, self).__init__()

        self.drop_path_rate = drop_rate
        self.num_features = self.embed_dim = embed_dim
        self.out_index = out_index

        self.patch_embed = HMLPStem(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            ParallelBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=0.0,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                parallel=parallel) for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(
            h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                         dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, x):
        x = self.prepare_tokens(x)

        features = list()
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                features.append(x)

        x = self.norm(x)
        last_feature = x[:, 0]
        last_feature = last_feature.unsqueeze(-1).unsqueeze(-1)
        features.append(last_feature)
        return features


@BACKBONES.register_module()
class ThreeThingTinyDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size, parallel):
        super(ThreeThingTinyDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=12,
            num_heads=3,
            parallel=parallel,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )


@BACKBONES.register_module()
class ThreeThingSmallDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size, parallel):
        super(ThreeThingSmallDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
            parallel=parallel,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )


@BACKBONES.register_module()
class ThreeThingBaseDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size, parallel):
        super(ThreeThingBaseDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            parallel=parallel,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
