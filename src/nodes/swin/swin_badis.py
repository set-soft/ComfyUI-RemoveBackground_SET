# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# Adapted by Salvador E. Tropea
#
# This variant of the Swin Transformer has some changes
# like:
# - Fixed size (224) so some stuff can be pre-computed
#   Like "atten_mask" attention masks
# - Has only 3 layers (not 4)
# - The downsampling progression is different.
# - No output normalization
#
# --------------------------------------------------------
import torch
import torch.nn as nn
from .swin_v1 import Mlp, to_2tuple, window_partition, window_reverse, WindowAttention, PatchMerging


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim * 4)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # Note: the size of inputs is fixed
            H = W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            atten_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            atten_mask = atten_mask.masked_fill(atten_mask != 0, float(-100.0)).masked_fill(atten_mask == 0, float(0.0))
        else:
            atten_mask = None

        self.register_buffer("atten_mask", atten_mask)

    def forward(self, x):
        H = W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.atten_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x += shortcut
        x += self.mlp(self.norm2(x))

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x, self.input_resolution, self.input_resolution)
        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.norm(x)
        return x


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=[2, 2, 18],
                 num_heads=[4, 8, 16],
                 window_size=7):
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

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.patch_embed(x)

        s = []
        s.append(x.view(b, int((x.shape[1])**0.5), int((x.shape[1])**(0.5)), -1).permute(0, 3, 1, 2).contiguous())
        for layer in self.layers:
            x = layer(x)
            s.append(x.view(b, int((x.shape[1])**0.5), int((x.shape[1])**(0.5)), -1).permute(0, 3, 1, 2).contiguous())

        return s
