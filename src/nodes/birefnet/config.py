from itertools import repeat
import collections.abc


class Config:
    def __init__(self, bb_index: int = 6) -> None:
        # Backbone settings
        # Only swin_v1_l and swin_v1_t used
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
            'swin_v1_t', 'swin_v1_s',               # 3, 4
            'swin_v1_b', 'swin_v1_l',               # 5-bs9, 6-bs4
            'pvt_v2_b0', 'pvt_v2_b1',               # 7, 8
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][bb_index]
        self.lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
            'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
        }[self.bb]
        self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-3:]

        # others
        self.device = [0, 'cpu'][0]     # .to(0) == .to('cuda:0')


# Copied and simplified implementation of to_2tuple.
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)
