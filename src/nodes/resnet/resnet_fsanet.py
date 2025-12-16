#
# ResNet: Deep Residual Learning for Image Recognition
# Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
# Microsoft Research {kahe, v-xiangz, v-shren, jiansun}@microsoft.com
# https://arxiv.org/pdf/1512.03385
#
# https://github.com/KaimingHe/deep-residual-networks/tree/master
#
# This code is the same found in Torch Vision implementation
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
#
# Another interesting implementation:
# https://github.com/tanjeffreyz/deep-residual-learning/blob/main/models.py
#
# This implementation is adapted to FSANet use.
#
import torch.nn as nn
from ..resnet.resnet import Bottleneck


class ResNet50(nn.Module):
    def __init__(self, mode='rgb'):
        self.inplanes = 64
        super().__init__()
        if (mode == 'rgb'):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif (mode == 'rgbf'):
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif (mode == "share"):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.conv1_d = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.inplanes = 512

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
