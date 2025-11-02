# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# Adapted by Salvador E. Tropea
#
# This variant of the Swin Transformer has some changes
# I constructed it from the "fixed" version
# - Normalization output is restored
# - Returns the layer outputs before downsampler
# - Is also fixed at 384
# - Has the v1 Base geometry
# --------------------------------------------------------
import torch.nn as nn
from .swin_v1 import PatchMerging
from .swin_fixed_size import PatchEmbed, BasicLayer as BasicLayerBase


class BasicLayer(BasicLayerBase):
    """ A basic Swin Transformer layer for one stage. """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, downsample=None):
        super().__init__(dim, input_resolution, depth, num_heads, window_size, downsample=downsample, save_atten_mask=False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x_ori = x

        if self.downsample is not None:
            x = self.downsample(x, self.input_resolution, self.input_resolution)
        return x, x_ori  # Here we also return the output without downsampler


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
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

        # build encoder and bottleneck layers
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

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (num_layers - 1)))

    # Encoder and Bottleneck
    def forward(self, x):
        x = self.patch_embed(x)

        x_no_downsample = []
        for layer in self.layers:
            x, x_ori = layer(x)
            x_no_downsample.append(x_ori)

        x = self.norm(x)  # B L C

        return x, x_no_downsample
