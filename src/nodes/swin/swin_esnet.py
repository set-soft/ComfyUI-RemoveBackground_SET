# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# Adapted by Salvador E. Tropea
#
# This variant of the Swin Transformer has some changes
# I constructed it from the "fixed" version
# - Is also fixed at 384
# - Has the v1 Base geometry
# - The returned features are for what ESNet needs
# --------------------------------------------------------
import torch.nn as nn
from ..swin.swin_v1 import PatchMerging
from ..swin.swin_fixed_size import PatchEmbed, BasicLayer


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 384
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 128
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 12
    """

    def __init__(self,
                 img_size=384,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18, 2],
                 num_heads=[4, 8, 16, 32],
                 window_size=12):
        super().__init__()
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = img_size // patch_size

        # build layers
        num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=patches_resolution // (2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               downsample=PatchMerging if (i_layer < num_layers - 1) else None)
            self.layers.append(layer)

    def resize_feat(self, x, num_passed, dim):
        sizes = [96, 48, 24, 12, 12]
        # if self.img_size==224:
        #     sizes = [56, 28, 14, 7, 7]
        size = sizes[num_passed]
        resize_x = x.view(-1, size, size, dim).permute(0, 3, 1, 2).contiguous()
        return resize_x

    def forward(self, x):
        x = self.patch_embed(x)

        features = [self.resize_feat(x, 0, self.layers[0].dim)]
        for num_passed, layer in enumerate(self.layers):
            features.append(self.resize_feat(x, num_passed, layer.dim))
            x = layer(x)

        return features
