import torch
import torch.nn as nn
import torch.nn.functional as F


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
