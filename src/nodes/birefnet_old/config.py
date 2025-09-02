from itertools import repeat
import collections.abc


class Config:
    def __init__(self) -> None:
        # Backbone
        # Only swin_v1_l used
        self.bb = 'swin_v1_l'
        self.lateral_channels_in_collection = [1536, 768, 384, 192]
        self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-3:]


# Copied and simplified implementation of to_2tuple.
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)
