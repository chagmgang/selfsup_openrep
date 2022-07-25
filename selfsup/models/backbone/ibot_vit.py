import numpy as np
import torch
import torch.nn as nn

from selfsup.models.backbone.dino_vit import \
    VisionTransformer as DinoVisionTransformer
from ..builder import BACKBONES


class iBOTVisionTransformer(DinoVisionTransformer):

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
        super(iBOTVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            out_index=out_index,
            **kwargs)

    def make_maskmap(self, batch, nc, ratio):
        maskmap = list()
        for _ in range(batch):
            num_mask_token = int(nc * ratio)
            single_maskmap = np.arange(nc)
            np.random.shuffle(single_maskmap)
            single_maskmap = np.where(single_maskmap < num_mask_token, True,
                                      False)
            maskmap.append(single_maskmap)
        return torch.as_tensor(maskmap)

    def prepare_tokens(self, x, mask_modeling=None, ratio=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        if ratio:
            maskmap = self.make_maskmap(x.shape[0], x.shape[1], ratio)
            maskmap = maskmap.to(x.device)
            x[maskmap] = mask_modeling.to(x.dtype)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)
        return x

    def forward(self, x, mask_modeling=None, ratio=None):
        x = self.prepare_tokens(x, mask_modeling, ratio)
        features = list()
        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.out_index:
                features.append(x)

        x = self.norm(x)
        features.append(x)

        return features


@BACKBONES.register_module()
class TinyiBOTViT(iBOTVisionTransformer):

    def __init__(self, img_size, patch_size):
        super(TinyiBOTViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=192,
            depth=6,
            num_heads=3)


@BACKBONES.register_module()
class SmalliBOTViT(iBOTVisionTransformer):

    def __init__(self, img_size, patch_size):
        super(SmalliBOTViT, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=384,
            depth=12,
            num_heads=6)
