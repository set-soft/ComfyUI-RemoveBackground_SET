#
# ESNet: Evolution and Succession Network for High-Resolution Salient Object Detection
#
# Hongyu Liu, Runmin Cong, Hua Li, Qianqian Xu, Qingming Huang, Wei Zhang
# https://openreview.net/pdf?id=SERrqPDvoY
#
# https://github.com/big-feather/ESNet_ICML24
#
# Adapted by Salvador E. Tropea
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..resnet.resnet import conv3x3
from ..resnet.resnet_badis import resnet50
from ..swin.swin_esnet import SwinTransformer
from .utils import get_gaussian_kernel
from ..ctdnet.ctdnet import CTDNet
from ..utils.misc import sigmoid_and_batched_min_max_norm


class revo(nn.Module):
    def __init__(self, k=3, sig=2):
        super().__init__()
        self.guass = get_gaussian_kernel(k, sig)  # Gaussian Filter

    def forward(self, f, s):
        # s * (f + f * (s - s * (s - self.guass(s)))) + f  Saliency-Guided Feature Refinement Module?
        se = s - torch.mul(s, s - self.guass(s))
        r = torch.mul(f + torch.mul(f, se), s) + f
        return r


class diff(nn.Module):
    def __init__(self, c1, c2, cn=1, up=True):
        super().__init__()
        self.conv_1 = conv3x3(c1, cn)
        self.bn_1 = nn.BatchNorm2d(cn)
        self.conv_2 = conv3x3(c2, cn)
        self.bn_2 = nn.BatchNorm2d(cn)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.out = conv3x3(cn, 1, bias=True)
        self.up = up

    def forward(self, f1, f2, s):
        f11 = self.conv_1(f1)
        f11 = self.bn_1(f11)
        f22 = self.conv_2(f2)
        f22 = self.bn_2(f22)
        if self.up:
            f22 = self.up2(f22)
            s = self.up2(s)
        f11 = torch.mul(f11, s)
        f22 = torch.mul(f22, s)
        o = self.out(f22 - torch.mul(f22, f11))
        return o


class CTDNet_Evo(CTDNet):
    def __init__(self, backbone, expansion=2, all_outputs=False, in_size=352):
        super().__init__(backbone, expansion=expansion, all_outputs=all_outputs)
        # The in_size and expansion are used outside the class
        self.in_size = in_size
        self.expansion = expansion

        self.r0 = revo(3, 1)
        self.r1 = revo(3, 1)
        self.r2 = revo(5, 2)
        self.r3 = revo(9, 4)
        self.r4 = revo(13, 7)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.d0 = diff(64, 64)
        self.d1 = diff(64, 64)
        self.d2 = diff(64, 64)
        self.d3 = diff(64, 64, up=False)

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        l1, l2, l3, l4, l5 = self.bkbone(x)

        path1_1 = F.avg_pool2d(l5, l5.size()[2:])
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=l5.size()[2:], mode='bilinear', align_corners=True)   # 1/32

        path1_2 = F.relu(self.path1_2(l5), inplace=True)                                            # 1/32
        tmp0 = self.fuse1_1(path1_1, path1_2)
        s5 = self.head_5(path1_1)
        path1_2 = self.r0(tmp0, s5.sigmoid())                                                       # 1/32
        path1_2 = F.interpolate(path1_2, size=l4.size()[2:], mode='bilinear', align_corners=True)   # 1/16

        path1_3 = F.relu(self.path1_3(l4), inplace=True)                                            # 1/16
        tmp1 = self.fuse1_2(path1_2, path1_3)                                                       # 1/16
        s4 = self.head_4(path1_2) + self.up2(s5) - self.d0(tmp1, tmp0, s5.sigmoid())
        path1 = self.r1(tmp1, s4.sigmoid())

        path2 = self.path2(l3)                                                                      # 1/8
        tmp2 = self.fuse12(path1, path2)                                                            # 1/8
        s3 = self.up2(self.head_3(path1)) + self.up2(s4) - self.d1(tmp2, tmp1, s4.sigmoid())
        path12 = self.r2(tmp2, s3.sigmoid())                                                        # 1/8
        path12 = F.interpolate(path12, size=l2.size()[2:], mode='bilinear', align_corners=True)     # 1/4

        path3_1 = F.relu(self.path3(l2), inplace=True)                                              # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)   # 1/4
        tmp3 = self.fuse3(path3_1, path3_2)
        s2 = self.head_2(path12) + self.up2(s3) - self.d2(tmp3, tmp2, s3.sigmoid())
        path3 = self.r3(tmp3, s2.sigmoid())                                                         # 1/4

        path_out = self.r4(self.fuse23(path12, path3), s2.sigmoid())                                # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        # logits_edge = F.interpolate(self.head_edge(tmp3), size=shape, mode='bilinear', align_corners=True)

        if self.all_outputs:
            return logits_1, l1, l2, path12, path1_2
        return sigmoid_and_batched_min_max_norm(logits_1)


def LrLM_resnet50(all_outputs=False):
    return CTDNet_Evo(resnet50(), expansion=4, all_outputs=all_outputs, in_size=352)


def LrLM_swin(all_outputs=False):
    return CTDNet_Evo(SwinTransformer(), all_outputs=all_outputs, in_size=384)


def ESNet_first(backbone_name, all_outputs=False):
    if backbone_name == 'resnet50':
        return LrLM_resnet50(all_outputs)
    if backbone_name == 'swin_v1_b':
        return LrLM_swin(all_outputs)
    raise ValueError("Unknown ESNet backbone")
