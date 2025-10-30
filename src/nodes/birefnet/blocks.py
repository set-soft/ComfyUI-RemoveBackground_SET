# ##################
# decoder_blocks.py
# ##################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64, old=False):
        super().__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        parallel_block_sizes = [3] if old else [1, 3, 7]
        self.dec_att = ASPPDeformable(in_channels=inter_channels, parallel_block_sizes=parallel_block_sizes)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.bn_in(x)
        x = self.relu_in(x)
        x = self.dec_att(x)
        x = self.conv_out(x)
        x = self.bn_out(x)
        return x


class OldDecBlk(BasicDecBlk):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super().__init__(in_channels, out_channels, inter_channels, old=True)


# ###########
# aspp.py
# ###########


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super(_ASPPModuleDeformable, self).__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=[1, 3, 7]):
        super(ASPPDeformable, self).__init__()
        self.down_scale = 1
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256 // self.down_scale

        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, conv_size, padding=int(conv_size//2))
            for conv_size in parallel_block_sizes])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(self.in_channelster),
                                             nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)), out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        xs = [x1]
        for aspp_deform in self.aspp_deforms:
            xs.append(aspp_deform(x))
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        xs.append(x5)
        x = torch.cat(xs, dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


# ##################
# deform_conv.py
# ##################


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super().__init__()

        assert type(kernel_size) is tuple or type(kernel_size) is int

        kernel_size = kernel_size if type(kernel_size) is tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) is tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = deform_conv2d(
            input=x,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=(self.padding, self.padding),
            mask=modulator,
            stride=self.stride,
        )
        return x


# ##################
# lateral_blocks.py
# ##################


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, inter_channels=64):
        super(BasicLatBlk, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv(x)
        return x
