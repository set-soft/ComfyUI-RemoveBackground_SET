from itertools import repeat
import collections.abc


class Config:
    def __init__(self, small=False) -> None:
        # Backbone settings
        # Only swin_v1_l and swin_v1_t used
        self.bb = 'swin_v1_t' if small else 'swin_v1_l'
        self.lateral_channels_in_collection = [1536, 768, 384, 192] if small else [3072, 1536, 768, 384]


# Copied and simplified implementation of to_2tuple.
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)
