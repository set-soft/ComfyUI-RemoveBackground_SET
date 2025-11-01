#
# Boundary-Aware Dichotomous Image Segmentation (v2)
# Haonan Tang, Shuhan Chen, Yang Liu, Shiyu Wang, Zeyu Chen & Xuelong Hu
#
# https://link.springer.com/article/10.1007/s00371-024-03295-5 (Paid service!)
#
# https://github.com/m0ho/Boundary-Aware-Dichotomous-Image-Segmentation
#
# License: ??
#
# Adapted by Salvador E. Tropea
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..resnet.resnet_badis import resnet18
from ..swin.swin_badis import SwinTransformer


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super().__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        # feature = super(_DenseAsppBlock, self).forward(_input)
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class multi_scale_aspp(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, channel):
        super().__init__()
        self.ASPP_3 = _DenseAsppBlock(input_num=channel, num1=channel * 2, num2=channel, dilation_rate=3,
                                      drop_out=0.1, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=channel * 2, num1=channel * 2, num2=channel,
                                      dilation_rate=6, drop_out=0.1, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=channel * 3, num1=channel * 2, num2=channel,
                                       dilation_rate=12, drop_out=0.1, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=channel * 4, num1=channel * 2, num2=channel,
                                       dilation_rate=18, drop_out=0.1, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=channel * 5, num1=channel * 2, num2=channel,
                                       dilation_rate=24, drop_out=0.1, bn_start=True)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel * 6, out_channels=channel, kernel_size=1, padding=0)
        )

    def forward(self, _input):
        # feature = super(_DenseAsppBlock, self).forward(_input)
        aspp3 = self.ASPP_3(_input)
        feature = torch.cat((aspp3, _input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)

        feature = torch.cat((aspp24, feature), dim=1)

        aspp_feat = self.classification(feature)

        return aspp_feat


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super().__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DB1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()
        self.squeeze1 = nn.Sequential(
                    nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
        self.squeeze2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=2, padding=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        z = self.squeeze2(self.squeeze1(x))
        return z, z


class DB2(nn.Module):
    def __init__(self, inplanes, outplanes):
        super().__init__()
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes+outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, z):
        z = F.interpolate(z, size=x.size()[2:], mode='bilinear', align_corners=True)
        p = self.conv(torch.cat((x, z), 1))
        sc = self.short_cut(z)
        p = p+sc
        p2 = self.conv2(p)
        p = p+p2
        return p, p


class SplitConvBlock(nn.Module):
    def __init__(self, channel, scales):
        super().__init__()
        self.scales = scales
        self.width = math.ceil(channel/scales)
        self.channel1 = self.width
        self.channel2 = self.width + self.channel1//2
        self.channel3 = self.width + self.channel2//2
        self.channel4 = self.width + self.channel3//2
        self.channel5 = self.width + self.channel4//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.channel1, self.channel1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.channel1),
            nn.PReLU()
        )
        if scales > 2:
            self.conv2 = nn.Sequential(
                nn.Conv2d(self.channel2, self.channel2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(self.channel2),
                nn.PReLU()
            )
        if scales > 3:
            self.conv3 = nn.Sequential(
                nn.Conv2d(self.channel3, self.channel3, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(self.channel3),
                nn.PReLU()
            )
        if scales > 4:
            self.conv4 = nn.Sequential(
                nn.Conv2d(self.channel4, self.channel4, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(self.channel4),
                nn.PReLU()
            )
        if scales > 5:
            self.conv5 = nn.Sequential(
                nn.Conv2d(self.channel5, self.channel5, kernel_size=3, stride=1, padding=8, dilation=8, bias=False),
                nn.BatchNorm2d(self.channel5),
                nn.PReLU()
            )

    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        sp1 = self.conv1(spx[0])

        if self.scales > 2:
            sp1x = torch.split(sp1, math.ceil(self.channel1/2), 1)
            sp2 = torch.cat((spx[1], sp1x[1]), 1)
            sp2 = self.conv2(sp2)

        if self.scales > 3:
            sp2x = torch.split(sp2, math.ceil(self.channel2/2), 1)
            sp3 = torch.cat((spx[2], sp2x[1]), 1)
            sp3 = self.conv3(sp3)

        if self.scales > 4:
            sp3x = torch.split(sp3, math.ceil(self.channel3/2), 1)
            sp4 = torch.cat((spx[3], sp3x[1]), 1)
            sp4 = self.conv4(sp4)

        if self.scales > 5:
            sp4x = torch.split(sp4, math.ceil(self.channel4/2), 1)
            sp5 = torch.cat((spx[4], sp4x[1]), 1)
            sp5 = self.conv5(sp5)

        if self.scales == 1:
            x = sp1
        elif self.scales == 2:
            x = torch.cat((sp1, spx[1]), 1)
        elif self.scales == 3:
            x = torch.cat((sp1x[0], sp2, spx[2]), 1)
        elif self.scales == 4:
            x = torch.cat((sp1x[0], sp2x[0], sp3, spx[3]), 1)
        elif self.scales == 5:
            x = torch.cat((sp1x[0], sp2x[0], sp3x[0], sp4, spx[4]), 1)
        elif self.scales == 6:
            x = torch.cat((sp1x[0], sp2x[0], sp3x[0], sp4x[0], sp5, spx[5]), 1)

        return x


class BA1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.db2 = DB2(64, 64)

        self.HSC = SplitConvBlock(64, 6)

        self.edge1 = nn.Conv2d(64, 1, 3, padding=1)

        self.edge2 = nn.Conv2d(64, 1, 3, padding=1)

        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(256+32, 64, kernel_size=3, stride=1, dilation=1, padding=1),
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
        sr = self.HSC(s+r)
        out, _ = self.db2(sr, up)
        e = self.edge1(sr)
        out = self.edge2(out)
        return out, e


class BA2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.convert = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
        )
        self.conve1 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.conve2 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.conve3 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.conve4 = nn.Sequential(
            nn.Conv2d(out_channel+1, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.edge1 = nn.Conv2d(out_channel, 1, 3, padding=1)
        self.edge2 = nn.Conv2d(out_channel, 1, 3, padding=1)
        self.convr = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU(),
            nn.Conv2d(out_channel, 1, 3, padding=1)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(out_channel), nn.PReLU()
        )

    def forward(self, x, y, e):
        x = self.convert(x)
        # xe = self.conve1(torch.cat((x, y), 1))
        a = -1*torch.sigmoid(y) + 1
        x = a.expand_as(x).mul(x)

        y1 = self.conve2(torch.cat((x, y), 1))
        e0 = self.conve3(torch.cat((x, y), 1))  # e0 is y1
        e1 = self.conve4(torch.cat((e0, e), 1))
        y2 = self.branch1(torch.cat((e1, y1), 1))
        e2 = self.branch2(torch.cat((e1, y2), 1))
        e = self.edge1(e2) + e
        y = self.edge2(y2) + y

        return y, e


class decoder_BG(nn.Module):
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

        self.GF = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
            )

        self.d1 = DB1(512, 64)
        self.d2 = DB2(512, 64)
        self.d3 = DB2(64, 64)
        self.d4 = BA1()
        self.d5 = BA2(128+32, 64)
        self.d6 = BA2(64+32, 32)

    def forward(self, s1, s2, s3, s4, r2, r3, r4, r5):
        r5 = F.interpolate(r5, size=s2.size()[2:], mode='bilinear', align_corners=True)
        s1 = F.interpolate(s1, size=r4.size()[2:], mode='bilinear', align_corners=True)

        s4_, _ = self.d1(s4)
        s3_, _ = self.d2(s3, s4_)

        s2_ = self.sqz_s2(s2)
        r5_ = self.sqz_r5(r5)

        graft_feature_r5 = self.GF(torch.cat((s2_, r5_), 1))
        graft_feature_r5_, _ = self.d3(graft_feature_r5, s3_)

        y4, e4 = self.d4(s1, r4, graft_feature_r5_)
        y4_3 = F.interpolate(y4, r3.size()[2:], mode='bilinear', align_corners=True)
        e4_3 = F.interpolate(e4, r3.size()[2:], mode='bilinear', align_corners=True)

        y3, e3 = self.d5(r3, y4_3, e4_3)
        e3_2 = F.interpolate(e3, r2.size()[2:], mode='bilinear', align_corners=True)
        y3_2 = F.interpolate(y3, r2.size()[2:], mode='bilinear', align_corners=True)

        y2, e2 = self.d6(r2, y3_2, e3_2)

        return y2, e2, y3, e3, y4, e4


class FPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.sqz512 = nn.Sequential(nn.Conv2d(512, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU())
        self.sqz256 = nn.Sequential(nn.Conv2d(256, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU())
        self.sqz128 = nn.Sequential(nn.Conv2d(128, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU())
        self.sqz64 = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU())
        self.sqz64_ = nn.Sequential(nn.Conv2d(64, 32, 3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.PReLU())

        # my_FPSA
        self.mhsa5 = MHSA(32, width=12, height=12, heads=4)
        self.daspp5_1 = multi_scale_aspp(64)
        self.selayer5 = SELayer(channel=32, reduction=16)
        self.conv_se5 = Triple_Conv(64, 32)

        self.mhsa4 = MHSA(32, width=24, height=24, heads=4)
        self.daspp4_1 = multi_scale_aspp(64)
        self.daspp4_2 = multi_scale_aspp(96)
        self.selayer4 = SELayer(channel=32, reduction=16)
        self.conv_se4 = Triple_Conv(96, 32)

        self.mhsa3 = MHSA(32, width=48, height=48, heads=4)
        self.daspp3_1 = multi_scale_aspp(64)
        self.daspp3_2 = multi_scale_aspp(96)
        self.selayer3 = SELayer(channel=32, reduction=16)
        self.conv_se3 = Triple_Conv(96, 32)

        self.mhsa2 = MHSA(32, width=48, height=48, heads=4)
        self.daspp2_1 = multi_scale_aspp(64)
        self.daspp2_2 = multi_scale_aspp(96)
        self.selayer2 = SELayer(channel=32, reduction=16)
        self.conv_se2 = Triple_Conv(96, 32)

    def forward(self, input_r2, input_r3, input_r4, input_r5):

        r2_shape = input_r2.size()[2:]
        r3_shape = input_r3.size()[2:]
        r4_shape = input_r4.size()[2:]
        # r5_shape = input_r5.size()[2:]

        r5_rc = self.sqz512(input_r5)
        r4_rc = self.sqz256(input_r4)
        r3_rc = self.sqz128(input_r3)
        r2_rc = self.sqz64(input_r2)

        r5_rc_rs = F.interpolate(r5_rc, size=(12, 12), mode='bilinear', align_corners=True)
        r4_rc_rs = F.interpolate(r4_rc, size=(24, 24), mode='bilinear', align_corners=True)
        r3_rc_rs = F.interpolate(r3_rc, size=(48, 48), mode='bilinear', align_corners=True)
        r2_rc_rs = F.interpolate(r2_rc, size=(48, 48), mode='bilinear', align_corners=True)

        r5_rc_rs_ = self.mhsa5(r5_rc_rs)
        r4_rc_rs_ = self.mhsa4(r4_rc_rs)
        r3_rc_rs_ = self.mhsa3(r3_rc_rs)
        r2_rc_rs_ = self.mhsa2(r2_rc_rs)

        r5_rc_rs_5_1 = self.daspp5_1(torch.cat((r5_rc_rs_, r5_rc_rs), 1))
        r5_rc_rs_se = self.conv_se5(r5_rc_rs_5_1)
        r5_rc_rs_SE = self.selayer5(r5_rc_rs_se)
        r5_rc_rs_r4 = F.interpolate(r5_rc_rs_SE, size=r4_rc_rs.size()[2:], mode='bilinear', align_corners=True)

        r4_rc_rs_4_1 = self.daspp4_1(torch.cat((r4_rc_rs_, r5_rc_rs_r4), 1))
        r4_rc_rs_4_2 = self.daspp4_2(torch.cat((r4_rc_rs_4_1, r4_rc_rs), 1))
        r4_rc_rs_se = self.conv_se4(r4_rc_rs_4_2)
        r4_rc_rs_SE = self.selayer4(r4_rc_rs_se)
        r4_rc_rs_r3 = F.interpolate(r4_rc_rs_SE, size=r3_rc_rs.size()[2:], mode='bilinear', align_corners=True)

        r3_rc_rs_3_1 = self.daspp3_1(torch.cat((r3_rc_rs_, r4_rc_rs_r3), 1))
        r3_rc_rs_3_2 = self.daspp3_2(torch.cat((r3_rc_rs_3_1, r3_rc_rs), 1))
        r3_rc_rs_se = self.conv_se3(r3_rc_rs_3_2)
        r3_rc_rs_SE = self.selayer3(r3_rc_rs_se)
        r3_rc_rs_r2 = F.interpolate(r3_rc_rs_SE, size=r2_rc_rs.size()[2:], mode='bilinear', align_corners=True)

        r2_rc_rs_2_1 = self.daspp2_1(torch.cat((r2_rc_rs_, r3_rc_rs_r2), 1))
        r2_rc_rs_2_2 = self.daspp2_2(torch.cat((r2_rc_rs_2_1, r2_rc_rs), 1))
        r2_rc_rs_se = self.conv_se2(r2_rc_rs_2_2)
        r2_rc_rs_SE = self.selayer2(r2_rc_rs_se)

        conv54 = F.interpolate(r4_rc_rs_SE, size=r4_shape, mode='bilinear', align_corners=True)
        conv543 = F.interpolate(r3_rc_rs_SE, size=r3_shape, mode='bilinear', align_corners=True)
        conv5432 = F.interpolate(r2_rc_rs_SE, size=r2_shape, mode='bilinear', align_corners=True)

        r4_ = torch.cat((conv54, input_r4), 1)
        r3_ = torch.cat((conv543, input_r3), 1)
        r2_ = torch.cat((conv5432, input_r2), 1)

        return r2_, r3_, r4_


class BADIS(nn.Module):
    def __init__(self):
        super().__init__()
        self.test_conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.decoder = decoder_BG()
        self.myFPSA = FPT()

        self.resnet = resnet18()
        self.swin = SwinTransformer()

    def forward(self, x):
        shape = x.size()[2:]

        r2, r3, r4, r5 = self.resnet(x)
        s1, s2, s3, s4 = self.swin(F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True))
        r2_, r3_, r4_ = self.myFPSA(r2, r3, r4, r5)
        y2, e2, y3, e3, y4, e4 = self.decoder(s1, s2, s3, s4, r2_, r3_, r4_, r5)

        y2 = F.interpolate(y2, size=shape, mode='bilinear')
        # e2 = F.interpolate(e2, size=shape, mode='bilinear')
        # y3 = F.interpolate(y3, size=shape, mode='bilinear')
        # e3 = F.interpolate(e3, size=shape, mode='bilinear')
        # y4 = F.interpolate(y4, size=shape, mode='bilinear')
        # e4 = F.interpolate(e4, size=shape, mode='bilinear')

        return y2.sigmoid()  # , [y2, y3, y4, e2, e3, e4]
