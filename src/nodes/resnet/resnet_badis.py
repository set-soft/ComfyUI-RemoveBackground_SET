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
# This implementation is adapted to BADIS use.
#
import torch.nn as nn
import torch.nn.functional as F
from .resnet import conv1x1, BasicBlock


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  Removed
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128, stride=2)
        self.layer3 = self._make_layer(256, stride=2)
        self.layer4 = self._make_layer(512, stride=2)
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  Removed
        # self.fc = nn.Linear(512, 1000)  Removed

    def _make_layer(self, planes, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        layers.append(BasicBlock(planes, planes))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out2 = self.layer1(F.relu(self.bn1(self.conv1(x)), inplace=True))
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5  # All layer outputs are returned


def resnet18():
    r"""ResNet-18 model from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return ResNet()
