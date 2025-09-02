import torch.nn as nn
from .aspp import ASPPDeformable


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
