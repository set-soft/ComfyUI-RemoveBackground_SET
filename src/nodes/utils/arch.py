# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-BiRefNet-SET
import math
import torch
from ..birefnet.birefnet import BiRefNet
from ..birefnet.birefnet_old import BiRefNet as OldBiRefNet
from ..ben.ben import BEN_Base
from ..inspyrenet.InSPyReNet import InSPyReNet_SwinB


UNWANTED_PREFIXES = ['module.', '_orig_mod.']


# This is needed for old models
def fix_state_dict(state_dict):
    """ Remove bogus prefixes from the keys in the state dict """
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in UNWANTED_PREFIXES:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        if prefix_length:
            state_dict[k[prefix_length:]] = state_dict.pop(k)


class RemBgArch(object):
    def __init__(self, state_dict, logger):
        super().__init__()
        self.ok = False
        self.bb_ok = False
        self.why = 'Not initialized'
        self.w = self.h = 1024  # Default size
        fix_state_dict(state_dict)
        bb_name = 'bb'

        # Determine the window size for the swin_v1 transformer
        tensor = state_dict.get(bb_name+'.layers.0.blocks.0.attn.relative_position_bias_table')
        if tensor is None:
            bb_name = 'backbone'
            tensor = state_dict.get(bb_name+'.layers.0.blocks.0.attn.relative_position_bias_table')
            if tensor is None:
                self.why = 'No relative position bias table'
                return
        window = (math.sqrt(tensor.shape[0]) + 1) / 2
        if window != int(window):
            self.why = "Wrong swin_v1 bias table size"
            return
        self.window_size = int(window)

        # Find layers (stages), depths and number of heads
        self.layers = 0
        self.depths = []
        self.num_heads = []
        while f'{bb_name}.layers.{self.layers}.blocks.0.attn.relative_position_bias_table' in state_dict:
            # How many heads?
            table = state_dict[f'{bb_name}.layers.{self.layers}.blocks.0.attn.relative_position_bias_table']
            self.num_heads.append(table.shape[-1])
            # Analyze the blocks for this layer
            blocks = 0
            while f'{bb_name}.layers.{self.layers}.blocks.{blocks}.norm1.weight' in state_dict:
                blocks += 1
            self.depths.append(blocks)
            # One more layer
            self.layers += 1

        tensor = state_dict.get(f'{bb_name}.patch_embed.proj.weight')
        if tensor is None:
            self.why = 'No PatchEmbed found'
            return
        self.embed_dim = tensor.shape[0]

        logger.debug(f"Embed dim={self.embed_dim} Layers {self.layers} Depths {self.depths} Num Heads {self.num_heads} "
                     f"Window size: {self.window_size}")

        # Check if this is one of the known back bones
        if self.matches(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12):
            self.bb = 'swin_v1_l'
            self.channels = [3072, 1536, 768, 384]
        elif self.matches(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7):
            self.bb = 'swin_v1_t'
            self.channels = [1536, 768, 384, 192]
        elif self.matches(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12):
            self.bb = 'swin_v1_b'
            self.channels = [1024, 512, 256, 128]
        else:
            self.why = 'unknown geometry'
            return
        self.bb_ok = True
        logger.debug(f"Model backbone: {self.bb}")

        # mean and standard deviation of the entire ImageNet dataset
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        if self.bb == 'swin_v1_b':
            # BEN and InSPyReNet
            assert bb_name == 'backbone'

            if 'output.0.weight' in state_dict:
                # BEN
                self.version = 1
                self.model_type = 'BEN'
                # The code from HuggingFace uses: @torch.autocast(device_type="cuda",dtype=torch.float16)
                self.dtype = torch.float16  # state_dict['output.0.weight'].dtype
            elif 'context1.branch0.conv.weight' in state_dict:
                # InSPyReNet
                self.version = 1
                self.model_type = 'InSPyReNet'
                self.dtype = state_dict['context1.branch0.conv.weight'].dtype
            else:
                self.why = 'Unknown Swin B variant model'
                return
        else:
            # BiRefNet
            # Try to figure out which version is this
            if 'decoder.ipt_blk1.conv1.weight' not in state_dict:
                self.why = 'Missing Input Injection Blocks'
                return
            self.dtype = state_dict['decoder.ipt_blk1.conv1.weight'].dtype
            if 'decoder.ipt_blk5.conv1.weight' in state_dict:
                self.version = 2
                # The ComfyUI_BiRefNet_ll nodes uses this for new models
                # self.img_mean = [0.5, 0.5, 0.5]
                # self.img_std = [1.0, 1.0, 1.0]
                # But I couldn't find any reference to it in the original code
            else:
                self.version = 1
            self.model_type = 'BiRefNet'

        self.ok = True
        logger.debug(f"Model type: {self.model_type}")

    def matches(self, embed_dim, depths, num_heads, window_size):
        return (embed_dim == self.embed_dim and self.depths == depths and self.num_heads == num_heads and
                self.window_size == window_size)

    def check(self):
        if not self.bb_ok:
            raise ValueError(f"Unknown backbone: {self.why}")
        if not self.ok:
            raise ValueError(f"Wrong architecture: {self.why}")

    def instantiate_model(self):
        if self.model_type == 'BEN':
            return BEN_Base()
        if self.model_type == 'InSPyReNet':
            return InSPyReNet_SwinB(depth=64, pretrained=False, base_size=[1024, 1024], threshold=None)
        return BiRefNet(self) if self.version == 2 else OldBiRefNet(self)
