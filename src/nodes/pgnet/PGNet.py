#
# Pyramid Grafting Network for One-Stage High Resolution Saliency Detection
# Chenxi Xie, Changqun Xia, Mingcan Ma, Zhirui Zhao, Xiaowu Chen, Jia Li
# https://arxiv.org/abs/2204.05041
#
# https://github.com/iCVTEAM/PGNet/
# https://huggingface.co/spaces/Tennineee/PDFNet/tree/main
#
# License: MIT
#
# Note by Salvador E. Tropea (SET):
# I removed training code and made use of my copy of Swin
# Also removed the use of CLI args, the parameters from there are only for training or tied to SwinB geometry
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..badis_v2.blocks import DB1, DB2
from ..resnet.resnet_badis import resnet18
from ..swin.swin_fixed_size import SwinTransformer


class Grafting(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        batch_size = x.shape[0]
        channel = x.shape[1]
        sc = x
        x = x.view(batch_size, channel, -1).permute(0, 2, 1)
        sc1 = x
        x = self.lnx(x)
        y = y.view(batch_size, channel, -1).permute(0, 2, 1)
        y = self.lny(y)

        B, N, C = x.shape
        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1]
        y_k = y_k[0]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = (x+sc1)

        x = x.permute(0, 2, 1)
        x = x.view(batch_size, channel, *sc.size()[2:])
        x = self.conv2(x)+x
        return x, self.act(self.bn(self.conv(attn+attn.transpose(-1, -2))))


class DB3(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.db2 = DB2(64, 64)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )

        self.sqz_s1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )

    def forward(self, s, r, up):
        up = F.interpolate(up, size=s.size()[2:], mode='bilinear', align_corners=True)
        s = self.sqz_s1(s)
        r = self.sqz_r4(r)
        sr = self.conv3x3(s+r)
        out, _ = self.db2(sr, up)
        return out, out


class decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sqz_s2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )
        self.sqz_r5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )

        self.GF = Grafting(64, num_heads=8)
        self.d1 = DB1(512, 64)
        self.d2 = DB2(512, 64)
        self.d3 = DB2(64, 64)
        self.d4 = DB3()
        self.d5 = DB2(128, 64)
        self.d6 = DB2(64, 64)

    def forward(self, s1, s2, s3, s4, r2, r3, r4, r5):
        r5 = F.interpolate(r5, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s1 = F.interpolate(s1, size=r4.size()[2:], mode='bilinear', align_corners=True)

        s4_, _ = self.d1(s4)
        s3_, _ = self.d2(s3, s4_)

        s2_ = self.sqz_s2(s2)
        r5_ = self.sqz_r5(r5)
        graft_feature_r5, cam = self.GF(r5_, s2_)

        graft_feature_r5_, _ = self.d3(graft_feature_r5, s3_)

        graft_feature_r4, _ = self.d4(s1, r4, graft_feature_r5_)

        r3_, _ = self.d5(r3, graft_feature_r4)

        r2_, _ = self.d6(r2, r3_)

        return r2_, cam, r5_, s2_


class PGNet(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.decoder = decoder()
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

        self.resnet = resnet18(with_maxpool=True)
        self.swin = SwinTransformer()

    def forward(self, x, shape=None, mask=None):
        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)

        r2, r3, r4, r5 = self.resnet(x)
        s1, s2, s3, s4 = self.swin(y)
        r2_, attmap, r5_, s2_ = self.decoder(s1, s2, s3, s4, r2, r3, r4, r5)

        # TODO: we scale here, then compute the sigmoid and outside we rescale again ... why not just once?
        pred1 = F.interpolate(self.linear1(r2_), size=shape, mode='bilinear')
        # wr = F.interpolate(self.linear2(r5_), size=(28, 28), mode='bilinear')
        # ws = F.interpolate(self.linear3(s2_), size=(28, 28), mode='bilinear')

        return torch.sigmoid(pred1)
        # return pred1, wr, ws, self.conv(attmap)   training values
