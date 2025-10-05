# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-BiRefNet-SET
import math
import os
import re
import torch
from seconohe.torch import model_to_target
try:
    import comfy.utils
    with_comfy = True
except ImportError:
    with_comfy = False

from ..birefnet.birefnet import BiRefNet
from ..birefnet.birefnet_old import BiRefNet as OldBiRefNet
from ..mvanet.mvanet import MVANet
from ..inspyrenet.InSPyReNet import InSPyReNet_SwinB
from ..u2net.u2net import U2NET_full, U2NET_lite, ISNet
from ..modnet.modnet import MODNet
from ..pdfnet.PDFNet import build_model as PDFNet

UNWANTED_PREFIXES = ['module.', '_orig_mod.',
                     # IS-Net anime-seg
                     'net.']
# The MCLM/MCRM in the code doesn't match the one really used
MVANET_MCLM_BUG = re.compile(r'(multifieldcrossatt\.(linear[12]|attention\.[567]))|'
                             r'(dec_blk\d\.(linear[12]|attention\.[4567]))')
MVANET_RENAME = {
    # They added 5 and 6, but the 1 and 2 were removed, the 5 is 1 and 6 is 2
    'multifieldcrossatt.linear5.weight': 'multifieldcrossatt.linear1.weight',
    'multifieldcrossatt.linear5.bias': 'multifieldcrossatt.linear1.bias',
    'multifieldcrossatt.linear6.weight': 'multifieldcrossatt.linear2.weight',
    'multifieldcrossatt.linear6.bias': 'multifieldcrossatt.linear2.bias',
}


# This is needed for old models
def fix_state_dict(state_dict):
    """ Remove bogus prefixes from the keys in the state dict """
    for k, v in list(state_dict.items()):
        # Remove IS-Net "Ground Truth Encoder", not for inference
        if k.startswith('gt_encoder.'):
            state_dict.pop(k)
            continue
        prefix_length = 0
        for unwanted_prefix in UNWANTED_PREFIXES:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        if prefix_length:
            state_dict[k[prefix_length:]] = state_dict.pop(k)


class RemBgArch(object):
    def __init__(self, state_dict, logger, fname):
        super().__init__()
        self.ok = False
        self.bb_ok = False
        self.why = 'Not initialized'
        self.w = self.h = 1024  # Default size
        self.needs_map = False
        self.version = 1
        self.logger = logger
        # mean and standard deviation of the entire ImageNet dataset
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        fix_state_dict(state_dict)

        # U-2-Net and IS-Net
        layer = "stage1.rebnconv1.conv_s1.weight"
        if layer in state_dict:
            self.bb = 'None'  # No backbone
            self.bb_ok = True

            if 'outconv.weight' in state_dict:
                # U2Net: u2net.pth and u2netp.pth
                # The output is fused
                self.w = self.h = 320
                self.model_type = 'U-2-Net'  # U-Square-Net
            elif 'conv_in.weight' in state_dict:
                # IS-Net isnet.pth and isnet-general-use.pth (i.e. BRIA RMBG v1.4)
                # Input with higher resolution, no fused output
                self.img_mean = [0.5, 0.5, 0.5]
                self.img_std = [1.0, 1.0, 1.0]
                # self.w = self.h = 1024  # Default size
                self.version = 2
                self.model_type = 'IS-Net'  # DIS
            else:
                self.why = 'Unknown U-2-Net implementation'
                return

            tensor = state_dict[layer]
            self.full = tensor.shape[0] == 32
            self.dtype = tensor.dtype
            self.ok = True
            logger.debug(f"Model type: {self.model_type} (Full: {self.full})")
            return

        # MODNet
        layer = 'lr_branch.backbone.model.features.0.0.weight'
        if layer in state_dict:
            if 'backbone.model.features.0.0.weight' in state_dict:
                self.bb_ok = True
                self.bb = 'mobilenetv2'
                logger.debug(f"Model backbone: {self.bb}")
            else:
                self.why = 'No model features'
                return
            self.model_type = 'MODNet'
            self.w = self.h = 512
            self.img_mean = [0.5, 0.5, 0.5]
            self.img_std = [0.5, 0.5, 0.5]
            self.dtype = state_dict[layer].dtype
            self.ok = True
            logger.debug(f"Model type: {self.model_type} ({self.bb}) [{self.dtype}]")
            return

        bb_name = 'bb'

        # Determine the window size for the swin_v1 transformer
        tensor = state_dict.get(bb_name+'.layers.0.blocks.0.attn.relative_position_bias_table')
        if tensor is None:
            bb_name = 'backbone'
            tensor = state_dict.get(bb_name+'.layers.0.blocks.0.attn.relative_position_bias_table')
            if tensor is None:
                bb_name = 'encoder'
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

        if self.bb == 'swin_v1_b':
            layer = 'decoder.FSE_mix.0.I_channelswich.0.weight'
            if layer in state_dict:
                # PDFNet
                assert bb_name == 'encoder'
                # No normalization applied
                self.img_mean = [0.0, 0.0, 0.0]
                self.img_std = [1.0, 1.0, 1.0]
                self.model_type = 'PDFNet'
                self.needs_map = True
                self.dtype = state_dict[layer].dtype
                # Remove training layers we don't use
                for k, v in list(state_dict.items()):
                    layer = k.split('.')[0]
                    if layer in {'IntegrityPriorLoss'}:
                        del state_dict[k]
            else:
                # BEN, InSPyReNet and MVANet
                assert bb_name == 'backbone'

                if 'output.0.weight' in state_dict:
                    # MVANet
                    self.model_type = 'MVANet'
                    if 'conv1.1.weight' in state_dict:
                        # MVANet original and messy
                        # Note: this difference is triggered by the use of BatchNorm2d instead of InstanceNorm2d in make_cbr
                        self.ben_variant = False
                        self.dtype = state_dict['conv1.1.weight'].dtype
                        if 'multifieldcrossatt.linear5.weight' in state_dict:
                            # Fix known bugs in available network
                            # Remove bogus layers
                            logger.debug('Removing bogus layers ...')
                            for k, v in list(state_dict.items()):
                                if MVANET_MCLM_BUG.match(k):
                                    logger.debug('- '+k)
                                    del state_dict[k]
                            # Rename some layers to match BEN numbering
                            # https://github.com/qianyu-dlut/MVANet/issues/3
                            for k, v in MVANET_RENAME.items():
                                state_dict[v] = state_dict.pop(k)
                    else:
                        # BEN
                        self.ben_variant = True
                        # The code from HuggingFace uses: @torch.autocast(device_type="cuda",dtype=torch.float16)
                        self.dtype = torch.float16  # state_dict['output.0.weight'].dtype
                    # Remove sideout layers, only used during training
                    if 'sideout5.0.weight' in state_dict:
                        for n in range(5):
                            del state_dict[f"sideout{n+1}.0.weight"]
                            del state_dict[f"sideout{n+1}.0.bias"]
                elif 'context1.branch0.conv.weight' in state_dict:
                    # InSPyReNet
                    self.model_type = 'InSPyReNet'
                    self.dtype = state_dict['context1.branch0.conv.weight'].dtype
                    # This information is in the YAML file, but this doesn't map to loading a standalone file
                    lower_case_fname = os.path.basename(fname).lower()
                    if 'fast' not in lower_case_fname:
                        if 'base' not in lower_case_fname:
                            logger.warning("Assuming a `base` InSPyReNet model, if `fast` please add it to the file name")
                        self.base_size = [1024, 1024]
                    else:
                        self.w = self.h = 384
                        self.base_size = [384, 384]
                    logger.debug(f"Using base size: {self.base_size}")
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

    def instantiate_model(self, state_dict, device, dtype):
        if self.model_type == 'MVANet':
            model = MVANet(ben_variant=self.ben_variant)
        elif self.model_type == 'InSPyReNet':
            model = InSPyReNet_SwinB(depth=64, base_size=self.base_size)
        elif self.model_type == 'BiRefNet':
            model = BiRefNet(self) if self.version == 2 else OldBiRefNet(self)
        elif self.model_type == 'U-2-Net':
            model = U2NET_full() if self.full else U2NET_lite()
        elif self.model_type == 'IS-Net':
            model = ISNet()
        elif self.model_type == 'MODNet':
            model = MODNet(backbone_arch=self.bb, backbone_pretrained=False)
        elif self.model_type == 'PDFNet':
            model = PDFNet(backbone_arch=self.bb)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = model
        self.target_device = self.model.target_device = torch.device(device)
        self.target_dtype = dtype
        model.load_state_dict(state_dict)
        model.to(dtype=dtype)
        model.eval()
        return model

    def _run_inference(self, image, depth):
        if self.needs_map:
            if depth.dim() == 3:
                depth = depth.unsqueeze(0)
            return self.model(image.to(self.target_device, dtype=self.target_dtype),
                              depth.to(self.target_device, dtype=self.target_dtype)).cpu().float()
        return self.model(image.to(self.target_device, dtype=self.target_dtype)).cpu().float()

    def run_inference(self, image_bchw, depth_bchw, batched):
        b = image_bchw.shape[0]
        with model_to_target(self.logger, self.model):
            if batched:
                mask_bchw = self._run_inference(image_bchw, depth_bchw)
            else:
                if with_comfy:
                    progress_bar_ui = comfy.utils.ProgressBar(b)
                _mask_bchw = []
                for n, each_image in enumerate(image_bchw):
                    _mask_bchw.append(self._run_inference(each_image.unsqueeze(0), depth_bchw[n]))
                    if with_comfy:
                        progress_bar_ui.update(1)

                mask_bchw = torch.cat(_mask_bchw, dim=0)  # (b, 1, h, w)
                del _mask_bchw

        mask_bhw = mask_bchw.squeeze(1)  # Discard the channels, which is 1 and we get (b, h, w)
        return mask_bhw
