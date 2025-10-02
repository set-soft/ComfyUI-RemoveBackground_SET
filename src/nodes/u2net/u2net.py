#
# U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection
#
# Xuebin Qin, Zichen Zhang, Chenyang Huang, Masood Dehghan, Osmar R. Zaiane and Martin Jagersand
# {xuebin,vincent.zhang,chuang8,masood1,zaiane,mj7}@ualberta.ca
# https://arxiv.org/pdf/2005.09007
#
# https://github.com/xuebinqin/U-2-Net
#
# Highly Accurate Dichotomous Image Segmentation
# Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, Luc Van Gool.
# https://arxiv.org/pdf/2203.03041
#
# https://github.com/xuebinqin/DIS
#
# License: Apache 2.0
#
# This code is the refactored U2Net code with support for ISNet variant of U2Net
# Changes by Salvador E. Tropea, assisted by Gemini Pro 2.5
#
import torch
import torch.nn as nn
import torch.nn.functional as F

import math

__all__ = ['U2NET_full', 'U2NET_lite', 'ISNet']


def _upsample_like(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def _size_map(x, height):
    # {height: size} for Upsample
    size = list(x.shape[-2:])
    sizes = {}
    for h in range(1, height):
        sizes[h] = size
        size = [math.ceil(w / 2) for w in size]
    return sizes


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dilate=1, stride=1):
        super().__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate, stride=stride)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))


class RSU(nn.Module):
    def __init__(self, name, height, in_ch, mid_ch, out_ch, dilated=False):
        super().__init__()
        self.name = name
        self.height = height
        self.dilated = dilated
        self._make_layers(height, in_ch, mid_ch, out_ch, dilated)

    def forward(self, x):
        sizes = _size_map(x, self.height)
        x = self.rebnconvin(x)

        # U-Net like symmetric encoder-decoder structure
        def unet(x, height=1):
            if height < self.height:
                x1 = getattr(self, f'rebnconv{height}')(x)
                if not self.dilated and height < self.height - 1:
                    x2 = unet(getattr(self, 'downsample')(x1), height + 1)
                else:
                    x2 = unet(x1, height + 1)

                x = getattr(self, f'rebnconv{height}d')(torch.cat((x2, x1), 1))
                return _upsample_like(x, sizes[height - 1]) if not self.dilated and height > 1 else x
            else:
                return getattr(self, f'rebnconv{height}')(x)

        return x + unet(x)

    def _make_layers(self, height, in_ch, mid_ch, out_ch, dilated=False):
        self.add_module('rebnconvin', REBNCONV(in_ch, out_ch))
        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.add_module('rebnconv1', REBNCONV(out_ch, mid_ch))
        self.add_module('rebnconv1d', REBNCONV(mid_ch * 2, out_ch))

        for i in range(2, height):
            dilate = 1 if not dilated else 2 ** (i - 1)
            self.add_module(f'rebnconv{i}', REBNCONV(mid_ch, mid_ch, dilate=dilate))
            self.add_module(f'rebnconv{i}d', REBNCONV(mid_ch * 2, mid_ch, dilate=dilate))

        dilate = 2 if not dilated else 2 ** (height - 1)
        self.add_module(f'rebnconv{height}', REBNCONV(mid_ch, mid_ch, dilate=dilate))


class U2NET(nn.Module):
    def __init__(self, cfgs, out_ch, version=1):
        super().__init__()
        self.out_ch = out_ch
        self.version = version
        self._make_layers(cfgs)

    def forward(self, x):
        # Keep a reference to the original input for final upsampling in ISNet
        original_x = x
        maps = []  # storage for maps

        if self.version == 2:
            # For ISNet, apply initial convolution that downsizes the input
            x = self.conv_in(x)

        # Calculate the size map based on the actual input to the U-Net stages
        sizes = _size_map(x, self.height)

        # --- Nested functions for recursive U-Net traversal ---

        # side saliency map
        def unet(x, height=1):
            # Base case: deepest layer
            if height == 6:
                x = getattr(self, f'stage{height}')(x)
                side(x, height)
                # Upsample to match the size of the corresponding encoder stage (stage5)
                return _upsample_like(x, sizes[height - 1])

            # Recursive step for encoder and decoder
            # 1. Pass through encoder stage
            x1 = getattr(self, f'stage{height}')(x)
            # 2. Go deeper into the network
            x2 = unet(getattr(self, 'downsample')(x1), height + 1)
            # 3. Concatenate and pass through decoder stage
            #    x2 is now correctly sized to match x1
            x_d = getattr(self, f'stage{height}d')(torch.cat((x2, x1), 1))
            side(x_d, height)
            # 4. Upsample for the next level up in the decoder
            if height > 1:
                return _upsample_like(x_d, sizes[height - 1])
            else:
                return x_d

        def side(x, h):
            # Generate side output saliency map (before sigmoid)
            x = getattr(self, f'side{h}')(x)

            # For ISNet, upsample side outputs to the original image size
            if self.version == 2:
                x = _upsample_like(x, original_x.shape[-2:])
            else:
                # For standard U2NET, upsample to the size of the main output
                x = _upsample_like(x, sizes[1])
            maps.append(x)

        def fuse():
            # Fuse the side output saliency maps
            maps.reverse()

            # ISNet returns only the first side output
            if self.version == 2:
                return torch.sigmoid(maps[0])

            # Standard U2NET concatenates all side outputs
            x = torch.cat(maps, 1)
            x = getattr(self, 'outconv')(x)
            # maps.insert(0, x)
            # return [torch.sigmoid(x) for x in maps]
            return torch.sigmoid(x)

        # --- Execute the forward pass ---
        unet(x)
        d = fuse()

        # Normalize the predicted SOD probability map
        ma = torch.max(d)
        mi = torch.min(d)
        # Add a small epsilon to avoid division by zero if ma and mi are the same
        dn = (d-mi)/(ma-mi+1e-8)

        return dn

    def _make_layers(self, cfgs):
        self.height = int((len(cfgs) + 1) / 2)
        if self.version == 2:
            self.add_module('conv_in', nn.Conv2d(3, 64, 3, stride=2, padding=1))

        self.add_module('downsample', nn.MaxPool2d(2, stride=2, ceil_mode=True))

        for k, v in cfgs.items():
            # Build RSU block
            self.add_module(k, RSU(v[0], *v[1]))
            if v[2] > 0:
                # Build side layer
                self.add_module(f'side{v[0][-1]}', nn.Conv2d(v[2], self.out_ch, 3, padding=1))

        if self.version == 1:
            # Build fuse layer only for the original U2NET
            self.add_module('outconv', nn.Conv2d(int(self.height * self.out_ch), self.out_ch, 1))


def U2NET_full():
    full = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 32, 64), -1],
        'stage2': ['En_2', (6, 64, 32, 128), -1],
        'stage3': ['En_3', (5, 128, 64, 256), -1],
        'stage4': ['En_4', (4, 256, 128, 512), -1],
        'stage5': ['En_5', (4, 512, 256, 512, True), -1],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=full, out_ch=1)


def U2NET_lite():
    lite = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 3, 16, 64), -1],
        'stage2': ['En_2', (6, 64, 16, 64), -1],
        'stage3': ['En_3', (5, 64, 16, 64), -1],
        'stage4': ['En_4', (4, 64, 16, 64), -1],
        'stage5': ['En_5', (4, 64, 16, 64, True), -1],
        'stage6': ['En_6', (4, 64, 16, 64, True), 64],
        'stage5d': ['De_5', (4, 128, 16, 64, True), 64],
        'stage4d': ['De_4', (4, 128, 16, 64), 64],
        'stage3d': ['De_3', (5, 128, 16, 64), 64],
        'stage2d': ['De_2', (6, 128, 16, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=lite, out_ch=1)


def ISNet():
    isnet = {
        # cfgs for building RSUs and sides
        # {stage : [name, (height(L), in_ch, mid_ch, out_ch, dilated), side]}
        'stage1': ['En_1', (7, 64, 32, 64), 64],
        'stage2': ['En_2', (6, 64, 32, 128), 64],
        'stage3': ['En_3', (5, 128, 64, 256), 128],
        'stage4': ['En_4', (4, 256, 128, 512), 256],
        'stage5': ['En_5', (4, 512, 256, 512, True), 512],
        'stage6': ['En_6', (4, 512, 256, 512, True), 512],
        'stage5d': ['De_5', (4, 1024, 256, 512, True), 512],
        'stage4d': ['De_4', (4, 1024, 128, 256), 256],
        'stage3d': ['De_3', (5, 512, 64, 128), 128],
        'stage2d': ['De_2', (6, 256, 32, 64), 64],
        'stage1d': ['De_1', (7, 128, 16, 64), 64],
    }
    return U2NET(cfgs=isnet, out_ch=1, version=2)
