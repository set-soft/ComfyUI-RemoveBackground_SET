#
# ESNet: Evolution and Succession Network for High-Resolution Salient Object Detection
#
# Hongyu Liu, Runmin Cong, Hua Li, Qianqian Xu, Qingming Huang, Wei Zhang
# https://openreview.net/pdf?id=SERrqPDvoY
#
# https://github.com/big-feather/ESNet_ICML24
#
# Author: Salvador E. Tropea
# This is the glue between ESNet_first and ESNet_second, based on the test example.
#
import torch.nn as nn
import torch.nn.functional as F
from .ESNet_first import ESNet_first
from .ESNet_second import ESNet_second
from ..utils.misc import sigmoid_and_batched_min_max_norm


class ESNet(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.first = ESNet_first(backbone_name, all_outputs=True)
        self.lr_size = self.first.in_size
        self.second = ESNet_second(self.first.expansion)
        # Helper to compute the edges
        self.pool = nn.MaxPool2d((7, 7), stride=1, padding=3)

    def forward(self, x):
        # First [evolution stage with Low-resolution Location Model (LrLM)]
        LR_image = F.interpolate(x, size=(self.lr_size, self.lr_size), mode='bilinear', align_corners=False)
        s, l1, l2, path12, path1_2 = self.first(LR_image)

        # Lr salient map
        s_map = sigmoid_and_batched_min_max_norm(s)

        # Extract the boundary or edges of the bright regions in the input map s
        ext_edge = self.pool(s_map) - s_map  # Dilated - Normal -> external
        int_edge = s_map - (-1 * self.pool(-1 * s_map))  # MinPool(x) = -MaxPool(-x), Normal - Eroded -> internal
        del s_map

        # External Boundary
        ext_edge = sigmoid_and_batched_min_max_norm(ext_edge)
        # Internal Boundary
        int_edge = sigmoid_and_batched_min_max_norm(int_edge)

        # Second [succession stage with Highresolution Refinement Model (HrRM)]
        # Note: the test scales the edges to 1280x1280, but the second stage scales them to the x.shape/4
        #       I see no point in going from Lr -> 1280x1280 -> input/4. I removed the 1280x1280 step
        logits = self.second(x, path12, path1_2, l1, l2, ext_edge, int_edge)

        # Normalization
        return sigmoid_and_batched_min_max_norm(logits)
