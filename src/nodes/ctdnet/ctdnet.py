#
# Complementary Trilateral Decoder for Fast and Accurate Salient Object Detection
#
# Zhao, Zhirui and Xia, Changqun and Xie, Chenxi and Li, Jia
# https://changqunxia.github.io/data/2021_MM_CTD.pdf
#
# https://github.com/zhaozhirui/CTDNet
#
# Adapted by Salvador E. Tropea
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..resnet.resnet import conv3x3, conv1x1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            x = self.downsample(x)

        return F.relu(out+x, inplace=True)


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)

        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5


# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out


# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        # left = F.relu(left_1 * right_1, inplace=True)
        # right = F.relu(left_2 * right_2, inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out


# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = conv3x3(2, 1)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out


# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv_atten = conv1x1(channel, channel)
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        # x = torch.cat([x_1, x_edge], dim=1)
        x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten) + x
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out


class CTDNet(nn.Module):
    def __init__(self, backbone, expansion=2, all_outputs=False):
        super().__init__()
        self.all_outputs = all_outputs
        self.bkbone = backbone

        self.path1_1 = nn.Sequential(conv1x1(512 * expansion, 64), nn.BatchNorm2d(64))
        self.path1_2 = nn.Sequential(conv1x1(512 * expansion, 64), nn.BatchNorm2d(64))
        self.path1_3 = nn.Sequential(conv1x1(256 * expansion, 64), nn.BatchNorm2d(64))

        self.path2 = SAM(128 * expansion, 64)

        self.path3 = nn.Sequential(conv1x1(64 * expansion, 64), nn.BatchNorm2d(64))

        self.fuse1_1 = FFM(64)
        self.fuse1_2 = FFM(64)
        self.fuse12 = CAM(64)
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)
        self.head_4 = conv3x3(64, 1, bias=True)
        self.head_5 = conv3x3(64, 1, bias=True)
        self.head_edge = conv3x3(64, 1, bias=True)

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        l1, l2, l3, l4, l5 = self.bkbone(x)

        path1_1 = F.avg_pool2d(l5, l5.size()[2:])
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=l5.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        path1_2 = F.relu(self.path1_2(l5), inplace=True)                                            # 1/32
        path1_2 = self.fuse1_1(path1_1, path1_2)                                                    # 1/32
        path1_2 = F.interpolate(path1_2, size=l4.size()[2:], mode='bilinear', align_corners=True)   # 1/16

        path1_3 = F.relu(self.path1_3(l4), inplace=True)                                            # 1/16
        path1 = self.fuse1_2(path1_2, path1_3)                                                      # 1/16
        # path1 = F.interpolate(path1, size=l3.size()[2:], mode='bilinear', align_corners=True)

        path2 = self.path2(l3)                                                                      # 1/8
        path12 = self.fuse12(path1, path2)                                                          # 1/8
        path12 = F.interpolate(path12, size=l2.size()[2:], mode='bilinear', align_corners=True)     # 1/4

        path3_1 = F.relu(self.path3(l2), inplace=True)                                              # 1/4
        path3_2 = F.interpolate(path1_2, size=l2.size()[2:], mode='bilinear', align_corners=True)   # 1/4
        path3 = self.fuse3(path3_1, path3_2)                                                        # 1/4

        path_out = self.fuse23(path12, path3)                                                       # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)

        if self.all_outputs:
            logits_2 = F.interpolate(self.head_2(path12), size=shape, mode='bilinear', align_corners=True)
            logits_3 = F.interpolate(self.head_3(path1), size=shape, mode='bilinear', align_corners=True)
            logits_4 = F.interpolate(self.head_4(path1_2), size=shape, mode='bilinear', align_corners=True)
            logits_5 = F.interpolate(self.head_5(path1_1), size=shape, mode='bilinear', align_corners=True)
            return logits_1, logits_edge, logits_2, logits_3, logits_4, logits_5
        return logits_1, logits_edge


def CTDNet_ResNet18():
    return CTDNet(ResNet(BasicBlock, [2, 2, 2, 2]), expansion=BasicBlock.expansion)
