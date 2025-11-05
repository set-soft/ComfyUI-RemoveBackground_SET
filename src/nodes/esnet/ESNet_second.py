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
import math
from einops import repeat
from .utils import get_sobel_kernel
from ..ctdnet.ctdnet import BRM, FFM
from ..resnet.resnet import conv3x3, conv1x1


class Bottleneck1(nn.Module):
    # Similar to ResNet.Bottleneck, but using 7x7
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=7, stride=stride, dilation=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


class decoder_end(nn.Module):
    def __init__(self, expansion=2, edge=False):
        super().__init__()
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)
        self.path3 = nn.Sequential(conv1x1(64 * expansion, 64), nn.BatchNorm2d(64))
        self.head_1 = conv3x3(64, 1, bias=True)

        self.edge = edge
        self.head_edge = conv3x3(64, 1, bias=True)

    def forward(self, path12, path1_2, l2, shape):
        path3_1 = F.relu(self.path3(l2), inplace=True)  # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)  # 1/4
        path3 = self.fuse3(path3_1, path3_2)  # 1/4

        path_out = self.fuse23(path12, path3)  # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        if self.edge:
            logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)
            return logits_1, logits_edge, path_out
        return logits_1


class evo_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, dilation=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1)
        self.sfem = SFEM()

    def make_layer(self, planes, blocks, stride):
        expansion = Bottleneck1.expansion
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * expansion))
        layers = [Bottleneck1(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, m1, m2):
        l1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        l1 = F.max_pool2d(l1, kernel_size=7, stride=2, padding=3)
        l1 = self.sfem(l1, m1, m2)
        l2 = self.layer1(l1)
        l2 = self.sfem(l2, m1, m2)
        return l2


class SFEM(nn.Module):
    """ Saliency Feature Enhancement Module """
    def __init__(self):
        super().__init__()

    def forward(self, f, m1, m2):
        B, C, H, W = f.shape
        v_s_sum = repeat(torch.sum(m1.reshape(B, H * W), dim=1, keepdim=True), 'b () -> b d', d=C)
        v_ns_sum = repeat(torch.sum(m2.reshape(B, H * W), dim=1, keepdim=True), 'b () -> b d', d=C)
        v_s = torch.sum(torch.mul(f, m1).reshape(B, C, H * W), dim=2) / (v_s_sum + 1e-8)
        v_ns = torch.sum(torch.mul(f, m2).reshape(B, C, H * W), dim=2) / (v_ns_sum + 1e-8)

        k = torch.cat([v_s.unsqueeze(-1), v_ns.unsqueeze(-1)], dim=-1)
        v = k.permute(0, 2, 1)

        f_m = f.permute(0, 2, 3, 1)
        q = f_m.reshape(B, H * W, C)
        norm_fact = 1 / math.sqrt(C)
        atten = nn.Softmax(dim=-1)(torch.bmm(q, k)) * norm_fact
        f_sa = torch.bmm(atten, v)
        f_sa = f_sa.reshape(B, H, W, C)

        return f + torch.mul(f_sa.permute(0, 3, 1, 2), m1+m2)


class sec_evo(nn.Module):
    def __init__(self, expansion):
        super().__init__()
        self.e1 = evo_encoder()

        self.d = decoder_end(expansion=expansion)  # edge=True
        self.d_diff = decoder_end(expansion=expansion)
        self.d_diff1 = decoder_end(expansion=expansion)

        self.so = get_sobel_kernel(256)
        self.conv = conv1x1(256, 64 * expansion)
        self.sfem = SFEM()

    def make_layer(self, planes, blocks, stride):
        expansion = Bottleneck1.expansion
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * expansion, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * expansion))
        layers = [Bottleneck1(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck1(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, path12, path1_2, l11, l22, mask, mask1):

        shape = (x.shape[-2]//4, x.shape[-1]//4)
        l2_hr = F.interpolate(l22, size=shape, mode='bilinear', align_corners=True)

        path12 = F.interpolate(path12, size=shape, mode='bilinear', align_corners=True)
        path1_2 = F.interpolate(path1_2, size=shape, mode='bilinear', align_corners=True)

        mask = F.interpolate(mask, size=shape, mode='bilinear', align_corners=False)
        mask1 = F.interpolate(mask1, size=shape, mode='bilinear', align_corners=False)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 1e-8
        mask1[mask1 > 0.5] = 1
        mask1[mask1 <= 0.5] = 1e-8
        l2_a = self.e1(x, mask, mask1)

        f_en = l2_hr + l2_hr * (self.conv(self.so(l2_a)).sigmoid())

        return self.d(path12, path1_2, f_en, x.shape[2:])


def ESNet_second(expansion):
    return sec_evo(expansion)
