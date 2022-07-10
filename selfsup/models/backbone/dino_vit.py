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


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
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

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer."""

    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 out_index=(3, 5, 7, 11),
                 **kwargs):
        super(VisionTransformer, self).__init__()

        self.num_features = self.embed_dim = embed_dim
        self.out_index = out_index

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)
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
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def prepare_tokens(self, x):
        B, nc ,w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

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
class TinyDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(TinyDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=12,
            num_heads=3,
        )


@BACKBONES.register_module()
class SmallDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(SmallDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6,
        )


@BACKBONES.register_module()
class BaseDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(BaseDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
        )


@BACKBONES.register_module()
class LargeDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(LargeDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )


@BACKBONES.register_module()
class HugeDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(HugeDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1280,
            depth=32,
            num_heads=16,
        )


@BACKBONES.register_module()
class GiantDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(GiantDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1408,
            mlp_ratio=48 / 11,
            depth=40,
            num_heads=16)


@BACKBONES.register_module()
class GiganticDinoViT(VisionTransformer):

    def __init__(self, img_size, patch_size):
        super(GiganticDinoViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1664,
            mlp_ratio=64 / 13,
            depth=48,
            num_heads=16,
        )
