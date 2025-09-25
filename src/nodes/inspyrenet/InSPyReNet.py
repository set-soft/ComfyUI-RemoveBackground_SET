#
# Revisiting Image Pyramid Structure for High Resolution Salient Object Detection (InSPyReNet)
#
# Taehun Kim, Kunhee Kim, Joonyeong Lee, Dongmin Cha, Jiho Lee, Daijin Kim
# arXiv:2209.09475
#
# https://github.com/plemeri/InSPyReNet
#
# License: MIT (was Apache 2.0)
#
# Note by Salvador E. Tropea (SET):
# I removed various training options and OpenCV dependency.
# Also made Transition and ImagePyramid be nn.Module
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.layers import ImagePyramid, Transition
from .modules.context_module import PAA_e
from .modules.attention_module import SICA
from .modules.decoder_module import PAA_d
from ..swin.swin_v1 import swin_v1_b


class InSPyReNet(nn.Module):
    def __init__(self, backbone, in_channels, depth=64, base_size=[384, 384]):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size

        self.context1 = PAA_e(self.in_channels[0], self.depth, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, stage=4)

        self.decoder = PAA_d(self.depth * 3, depth=self.depth, stage=2)

        self.attention0 = SICA(self.depth, depth=self.depth, base_size=self.base_size, stage=0, lmap_in=True)
        self.attention1 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=1, lmap_in=True)
        self.attention2 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=2)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')

        self.image_pyramid = ImagePyramid()  # 7, 1, 1

        self.transition0 = Transition(17)
        self.transition1 = Transition(9)
        self.transition2 = Transition(5)

    def forward_inspyre(self, x):
        B, _, H, W = x.shape

        x1, x2, x3, x4, x5 = self.backbone(x)

        x1 = self.context1(x1)  # 4
        x2 = self.context2(x2)  # 4
        x3 = self.context3(x3)  # 8
        x4 = self.context4(x4)  # 16
        x5 = self.context5(x5)  # 32

        f3, d3 = self.decoder([x3, x4, x5])  # 16

        f3 = self.res(f3, (H // 4,  W // 4))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3.detach())
        d2 = self.image_pyramid.reconstruct(d3.detach(), p2)  # 4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2.detach(), p2.detach())  # 2
        d1 = self.image_pyramid.reconstruct(d2.detach(), p1)  # 2

        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1.detach(), p1.detach())  # 2
        d0 = self.image_pyramid.reconstruct(d1.detach(), p0)  # 2

        # out = dict()
        # out['saliency'] = [d3, d2, d1, d0]
        # out['laplacian'] = [p2, p1, p0]

        return d0

    def forward(self, img, img_lr=None):  # forward_inference
        pred = torch.sigmoid(self.forward_inspyre(img))
        # Force the probabilities into a perfect [0, 1] range
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        return pred


def InSPyReNet_SwinB(depth, base_size):
    return InSPyReNet(swin_v1_b(), [128, 128, 256, 512, 1024], depth, base_size)
