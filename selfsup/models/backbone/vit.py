import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ..builder import BACKBONES

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    PreNorm(
                        dim,
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                ]))

    def forward(self, x):
        features = list()
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            features.append(x)
        return features


class ViT(nn.Module):

    def __init__(
        self,
        image_size,
        patch_size,
        dim,  # layer in/out dim
        depth,  # number of layer
        heads,  # number of head
        mlp_dim,  # hidden dim of ffn
        dim_head,  # dim of mhsa
        channels=3,
        dropout=0.,
        emb_dropout=0.,
        out_index=(3, 5, 7, 11)):  # noqa: E125
        super(ViT, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (
            image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                p1=patch_height,
                p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       dropout)
        self.out_index = out_index

    def forward(self, x):

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        features = self.transformer(x)
        out_features = list()
        for idx, feature in enumerate(features):
            if idx in self.out_index:
                out_features.append(feature)

        last_feature = features[-1][:, 0]
        last_feature = last_feature.unsqueeze(-1).unsqueeze(-1)

        out_features.append(last_feature)
        return out_features


@BACKBONES.register_module()
class ViTSmall(ViT):

    def __init__(self, image_size, patch_size):
        super(ViTSmall, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            dim=384,
            depth=12,
            heads=12,
            mlp_dim=1536,
            dim_head=64,
        )


@BACKBONES.register_module()
class ViTBase(ViT):

    def __init__(self, image_size, patch_size):
        super(ViTBase, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dim_head=64,
        )


@BACKBONES.register_module()
class ViTLarge(ViT):

    def __init__(self, image_size, patch_size):
        super(ViTBase, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            dim=1024,
            depth=24,
            heads=16,
            mlp_dim=4096,
            dim_head=64,
        )


@BACKBONES.register_module()
class ViTHuge(ViT):

    def __init__(self, image_size, patch_size):
        super(ViTBase, self).__init__(
            image_size=image_size,
            patch_size=patch_size,
            dim=1280,
            depth=32,
            heads=16,
            mlp_dim=5120,
            dim_head=64,
        )
