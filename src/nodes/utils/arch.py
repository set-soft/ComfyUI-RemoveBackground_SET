# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
from contextlib import nullcontext
import math
import os
import re
import safetensors.torch
import torch
from torchvision import transforms
from seconohe.bti import BatchedTensorIterator
from seconohe.torch import model_to_target, TorchProfile
from seconohe.logger import get_debug_level
try:
    from comfy.utils import common_upscale
    WITH_CONFY = True
except ImportError:
    WITH_CONFY = False
    import torch.nn.functional as F

    def common_upscale(img, w, h, method, crop):
        assert crop == "disabled"
        F.interpolate(img, size=(h, w), mode=method)


from ..birefnet.birefnet import BiRefNet
from ..birefnet.birefnet_old import BiRefNet as OldBiRefNet
from ..mvanet.mvanet import MVANet
from ..mvanet.finegrain_names import finegrain_convert
from ..inspyrenet.InSPyReNet import InSPyReNet_SwinB, InSPyReNet_Res2Net50
from ..u2net.u2net import U2NET_full, U2NET_lite, ISNet
from ..modnet.modnet import MODNet
from ..pdfnet.PDFNet import build_model as PDFNet
from ..diffdis.diffdis_pipeline import DiffDISPipeline, DiffDIS
from ..nodes_dan import DownloadAndLoadDepthAnythingV2Model, BASE_MODEL_NAME, DepthAnything_V2
from .. import DEFAULT_UPSCALE

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
FINEGRAIN_SWIN_KEY = ('SwinTransformer.Chain_1.BasicLayer.SwinTransformerBlock_1.Residual_1.WindowAttention.WindowSDPA.'
                      'rpb.relative_position_bias_table')


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


class ImagePreprocessor:
    def __init__(self, img_mean, img_std, resolution, upscale_method) -> None:
        interpolation = transforms.InterpolationMode(upscale_method)
        self.transform_image = transforms.Compose([transforms.Resize(resolution, interpolation=interpolation),
                                                   # output[channel] = (input[channel] - mean[channel]) / std[channel]
                                                   transforms.Normalize(img_mean, img_std)])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


def scale_image(img, w, h, method):
    if method == 'lanczos' and img.shape[1] == 1:
        # Lanczos is only implemented for RGB, not masks/depths
        method = 'bicubic'
    return common_upscale(img, w, h, method, "disabled")


def filter_mask(mask, threshold=4e-3):
    mask_binary = mask > threshold
    filtered_mask = mask * mask_binary
    return filtered_mask


class RemBg(object):
    def __init__(self, state_dict, logger, fname, vae=None, positive=None):
        super().__init__()
        self.ok = False
        self.bb_ok = False
        self.why = 'Not initialized'
        self.w = self.h = 1024  # Default size
        self.needs_map = False
        self.version = 1
        self.logger = logger
        self.vae = vae
        self.positive = positive
        self.da_model = None
        self.sub_type = None
        # mean and standard deviation of the entire ImageNet dataset
        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]
        lower_case_fname = os.path.basename(fname).lower()

        fix_state_dict(state_dict)

        # DiffDIS
        layer = "mid_block.attentions.0.transformer_blocks.0.attn3.to_q.weight"
        tensor = state_dict.get(layer)
        if tensor is None:
            layer = 'unet.'+layer
            tensor = state_dict.get(layer)
        if tensor is not None:
            self.bb = 'None'  # No backbone
            self.bb_ok = True
            self.img_mean = [0.5, 0.5, 0.5]
            self.img_std = [0.5, 0.5, 0.5]
            self.model_type = 'DiffDIS'
            self.dtype = tensor.dtype
            self.ok = True
            return

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

        #
        # Models with Swin/Res2Net as backbone
        #
        bb_name = self.is_res2net(['backbone'], state_dict)
        if bb_name is None:
            return

        if not bb_name:
            # Try Swin Transformer
            # Finegrain version of MVANet has a heavy rename
            if FINEGRAIN_SWIN_KEY in state_dict:
                state_dict = finegrain_convert(state_dict)

            bb_name = self.is_swin(['bb', 'backbone', 'encoder'], state_dict)
            if bb_name is None:
                return
            if not bb_name:
                self.why = "Unknown backbone"
                return

        logger.debug(f"Model backbone: {self.bb}")

        if self.bb == 'res2net50_v1b_26w_4s':
            if not self.is_inspyrenet(state_dict, lower_case_fname):
                self.why = 'Unknown Res2Net variant model'
                return
        elif self.bb == 'swin_v1_b':
            if not self.is_pdfnet(state_dict):
                # BEN, InSPyReNet and MVANet?
                if not self.is_mvanet(state_dict) and not self.is_inspyrenet(state_dict, lower_case_fname):
                    # Don't know about it
                    self.why = 'Unknown Swin B variant model'
                    return
                assert self.bb_prefix == 'backbone'
        else:  # swin_v1_l or swin_v1_t
            if not self.is_birefnet(state_dict):
                # Don't know about it
                self.why = 'Unknown Swin variant model'
                return

        self.ok = True
        logger.debug(f"Model type: {self.model_type}")

    def matches(self, embed_dim, depths, num_heads, window_size):
        return (embed_dim == self.embed_dim and self.depths == depths and self.num_heads == num_heads and
                self.window_size == window_size)

    # InSPyReNet
    def is_inspyrenet(self, state_dict, lower_case_fname):
        """ Check this is a InSPyReNet model """
        if 'context1.branch0.conv.weight' not in state_dict:
            return False
        self.model_type = 'InSPyReNet'
        self.dtype = state_dict['context1.branch0.conv.weight'].dtype
        # This information is in the YAML file, but this doesn't map to loading a standalone file
        if self.bb == 'res2net50_v1b_26w_4s' or 'fast' in lower_case_fname:
            self.w = self.h = 384
            self.base_size = [384, 384]
        else:
            if 'base' not in lower_case_fname:
                self.logger.warning("Assuming a `base` InSPyReNet model, if `fast` please add it to the file name")
            self.base_size = [1024, 1024]
        self.logger.debug(f"Using base size: {self.base_size}")
        return True

    # PDFNet
    def is_pdfnet(self, state_dict):
        tensor = state_dict.get('decoder.FSE_mix.0.I_channelswich.0.weight')
        if tensor is None:
            return False
        assert self.bb_prefix == 'encoder'
        # No normalization applied
        self.img_mean = [0.0, 0.0, 0.0]
        self.img_std = [1.0, 1.0, 1.0]
        self.model_type = 'PDFNet'
        self.needs_map = True
        self.dtype = tensor.dtype
        # Remove training layers we don't use
        for k, v in list(state_dict.items()):
            layer = k.split('.')[0]
            if layer in {'IntegrityPriorLoss'}:
                del state_dict[k]
        return True

    # MVANet/BEN
    def is_mvanet(self, state_dict):
        """ Check if this is an MVANet model, includes BEN """
        if 'output.0.weight' not in state_dict:
            return False
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
                self.logger.debug('Removing bogus layers ...')
                for k, v in list(state_dict.items()):
                    if MVANET_MCLM_BUG.match(k):
                        self.logger.debug('- '+k)
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
        return True

    # BiRefNet
    def is_birefnet(self, state_dict):
        """ Check if this is a BiRefNet model """
        # Try to figure out which version is this
        if 'decoder.ipt_blk1.conv1.weight' not in state_dict:
            return False
        self.dtype = state_dict['decoder.ipt_blk1.conv1.weight'].dtype
        if 'decoder.ipt_blk5.conv1.weight' in state_dict:
            self.version = 2
            # The ComfyUI_BiRefNet_ll nodes uses this for new models
            # self.img_mean = [0.5, 0.5, 0.5]
            # self.img_std = [1.0, 1.0, 1.0]
            # But I couldn't find any reference to it in the original code
        self.model_type = 'BiRefNet'
        return True

    def is_swin(self, names, state_dict):
        """ Do we have a Swin Transformer backbone? """
        for n in names:
            tensor = state_dict.get(n+'.layers.0.blocks.0.attn.relative_position_bias_table')
            if tensor is not None:
                break
        else:
            return False

        self.bb_prefix = bb_name = n
        # Determine the window size for the swin_v1 transformer
        window = (math.sqrt(tensor.shape[0]) + 1) / 2
        if window != int(window):
            self.why = "Wrong swin_v1 bias table size"
            return None
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
            return None
        self.embed_dim = tensor.shape[0]

        self.logger.debug(f"Embed dim={self.embed_dim} Layers {self.layers} Depths {self.depths} Num Heads {self.num_heads} "
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
            return None
        self.bb_ok = True
        return True

    def is_res2net(self, names, state_dict):
        for n in names:
            tensor = state_dict.get(n+".layer1.0.bns.0.weight")
            if tensor is not None:
                base_width = tensor.shape[0]
                break
        else:
            return False

        self.bb_prefix = n
        self.base_width = base_width
        self.layers = 0
        self.layer_blocks = []
        while f'{n}.layer{self.layers+1}.0.conv1.weight' in state_dict:
            # Analyze the blocks for this layer
            blocks = 0
            while f'{n}.layer{self.layers+1}.{blocks}.conv1.weight' in state_dict:
                blocks += 1
            self.layer_blocks.append(blocks)
            # One more layer
            self.layers += 1
        if not self.layers:
            return False

        tensor = state_dict.get("backbone.layer1.0.bn1.weight")
        if tensor is None:
            return False
        self.scale = tensor.shape[0] // base_width

        self.logger.debug(f"Res2Net: Base width: {base_width} Layers: {self.layer_blocks} Scale: {self.scale}")

        if self.matches_r2n(base_width=26, layers=[3, 4, 6, 3], scale=4):
            self.bb = 'res2net50_v1b_26w_4s'
        else:
            self.why = 'unknown geometry'
            return None

        self.bb_ok = True
        return True

    def matches_r2n(self, base_width, layers, scale):
        return self.base_width == base_width and layers == self.layer_blocks and self.scale == scale

    def check(self):
        if not self.bb_ok:
            raise ValueError(f"Unknown backbone: {self.why}")
        if not self.ok:
            raise ValueError(f"Wrong architecture: {self.why}")
        if self.model_type == 'DiffDIS':
            # This is a modified SD Turbo diffuser
            if self.vae is None:
                raise ValueError("DiffDIS models needs the SD-Turbo VAE")
            if self.positive is not None:
                # The model uses just 2 of the 77
                self.positive = self.positive[0][0][:, :2, :]
            else:
                # Load a pre-computed copy
                fname = os.path.join(os.path.dirname(__file__), '..', 'diffdis', 'positive.safetensors')
                self.logger.debug(f"Loading empty positive embeddings from `{fname}`")
                pos_dict = safetensors.torch.load_file(fname, device="cpu")
                self.positive = pos_dict['positive']

    def get_name(self):
        name = self.model_type
        if self.sub_type:
            name += " "+self.sub_type
        return name

    def instantiate_model(self, state_dict, device="cpu", dtype=torch.float32):
        if self.model_type == 'MVANet':
            model = MVANet(ben_variant=self.ben_variant)
        elif self.model_type == 'InSPyReNet':
            if self.bb == 'swin_v1_b':
                model = InSPyReNet_SwinB(depth=64, base_size=self.base_size)
            else:  # 'res2net50_v1b_26w_4s'
                model = InSPyReNet_Res2Net50(depth=64, base_size=self.base_size)
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
        elif self.model_type == 'DiffDIS':
            model = DiffDISPipeline(vae=self.vae, unet=DiffDIS(), positive=self.positive)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model = model
        self.target_device = self.model.target_device = torch.device(device)
        self.target_dtype = dtype
        model.load_state_dict(state_dict)
        model.to(dtype=dtype)
        model.eval()
        return model

    def show_inference_info(self, image, depths):
        # Some debug information about what we are doing
        logger = self.logger
        debug_level = get_debug_level(logger)
        if debug_level >= 1:
            logger.debug(f"Starting Remove Background inference: {self.model.__class__.__name__}")
            if debug_level >= 2:
                logger.debug(f"- Model: {self.model.target_device}/{self.target_dtype} with_edges {self.with_edges}")
                logger.debug(f"- Input: {image.shape} {image.device}/{image.dtype} "
                             f"iterations: {len(self.batched_iterator)}")
                if depths is None:
                    logger.debug("- Depths: None")
                else:
                    logger.debug(f"- Depths: {depths.shape} {depths.device}/{depths.dtype}")
                logger.debug(f"- Output: cpu/{self.out_dtype}")

    def init_depths(self, depths_bhw, batch_size, keep_depths):
        self.keep_depths = keep_depths
        if not self.needs_map:
            self.depths_bhw = depths_bhw
            self.depths_bchw = None
            self.create_depths = False
            return
        self.create_depths = depths_bhw is None
        if self.create_depths:
            if self.da_model is None:
                self.da_model = DownloadAndLoadDepthAnythingV2Model.execute(BASE_MODEL_NAME)[0]
                # Make the model work in the same device and with the same dtype as the main model
                model = self.da_model['model']
                model.target_dtype = self.target_dtype
                model.target_device = self.target_device
            if keep_depths:
                self.depths_bhw_list = []
        else:
            b = depths_bhw.shape[0]
            if b != self.img_b:
                raise ValueError(f"Found {self.img_b} images and {b} depths, provide the same amount")
            h, w = depths_bhw.shape[-2:]
            if self.img_h != h or self.img_w != w:
                raise ValueError(f"Images using {self.img_w}x{self.img_h} and depths using {w}x{h}, must be of the same size")
            self.depths_bchw = depths_bhw.unsqueeze(1)
            self.depths_bhw = depths_bhw

    def scale_to_model(self, img_bchw):
        if not self.needs_scale:
            return img_bchw
        return scale_image(img_bchw, self.model_w, self.model_h, self.scale_method)

    def scale_to_source(self, img_bchw):
        if not self.needs_scale:
            return img_bchw
        return scale_image(img_bchw, self.img_w, self.img_h, self.scale_method).clamp(0, 1)

    def get_depths(self, batch_range, images_bchw):
        if not self.needs_map:
            return None
        if not self.create_depths:
            return self.scale_to_model(self.batched_iterator.get_aux_batch(self.depths_bchw, batch_range))
        # Needed and not provided
        depths_bchw = DepthAnything_V2.process_low(self.da_model, images_bchw, images_bchw.shape[0],
                                                   out_dtype=self.target_dtype, out_device=self.target_device)
        self.collect_depths(depths_bchw)
        return depths_bchw

    def collect_depths(self, depths_bchw):
        if not self.keep_depths:
            return
        # Keep a copy to return
        self.depths_bhw_list.append(self.scale_to_source(depths_bchw).squeeze(1).to(device="cpu", dtype=self.out_dtype))

    def get_all_depths(self):
        if not self.keep_depths:
            return None
        if not self.needs_map or not self.create_depths:
            depths_bhw = self.depths_bhw
            del self.depths_bhw
            del self.depths_bchw
            if depths_bhw is None:
                # We want depths, but we don't have it, create small dummies
                return torch.zeros((self.img_b, 64, 64), dtype=self.out_dtype, device="cpu")
            return depths_bhw
        # The ones we created and collected
        depths_bhw = torch.cat(self.depths_bhw_list, dim=0)
        del self.depths_bhw_list
        return depths_bhw

    def get_depths_context(self):
        return model_to_target(self.logger, self.da_model['model']) if self.create_depths else nullcontext()

    def init_masks(self, keep_masks, mask_threshold):
        self.keep_masks = keep_masks
        self.mask_threshold = mask_threshold
        if keep_masks:
            self.masks_bhw_list = []

    def collect_masks(self, masks_bchw):
        if not self.keep_masks:
            # Not collecting them, but they are needed to compute the output images
            return self.scale_to_source(masks_bchw) if self.gen_outs else None
        # Keep a copy to return
        masks_bchw_scaled = self.scale_to_source(masks_bchw)
        self.masks_bhw_list.append(masks_bchw_scaled.squeeze(1).to(device="cpu", dtype=self.out_dtype))
        return masks_bchw_scaled if self.gen_outs else None

    def get_masks(self):
        if not self.keep_masks:
            return None
        masks_bhw = torch.cat(self.masks_bhw_list, dim=0)
        del self.masks_bhw_list
        return masks_bhw

    def init_edges(self, keep_edges):
        self.keep_edges = keep_edges
        self.with_edges = hasattr(self.model, 'edges')
        if keep_edges and self.with_edges:
            self.edges_bhw_list = []

    def collect_edges(self):
        if not self.with_edges:
            return
        if self.keep_edges:
            self.edges_bhw_list.append(self.scale_to_source(self.model.edges).to(device="cpu", dtype=self.out_dtype))
        del self.model.edges

    def get_edges(self):
        if not self.keep_edges:
            return None
        if self.with_edges:
            # We have them
            edges_bhw = torch.cat(self.edges_bhw_list, dim=0)
            del self.edges_bhw_list
            return edges_bhw
        # We want edges, but we don't have it, create small dummies
        return torch.zeros((self.img_b, 64, 64), dtype=self.out_dtype, device="cpu")

    def init_images(self, images_bhwc, batch_size, preproc_img, model_w, model_h, scale_method, out_dtype):
        b, h, w, c = images_bhwc.shape
        self.img_b = b
        self.img_w = w
        self.img_h = h
        self.img_c = c
        self.out_dtype = images_bhwc.dtype if out_dtype is None else out_dtype
        # Optional image scale
        self.model_w = model_w or w
        self.model_h = model_h or h
        self.scale_method = scale_method
        self.needs_scale = preproc_img
        # TODO: more elaborate check depending on the model
        img_w = self.model_w if preproc_img else w
        img_h = self.model_h if preproc_img else h
        if img_h % 32 or img_w % 32:
            raise ValueError(f"Image size must be a multiple of 32 (not {img_w}x{img_h})")
        if preproc_img:
            self.image_preproc = ImagePreprocessor(self.img_mean, self.img_std, resolution=(self.model_h, self.model_w),
                                                   upscale_method=scale_method)
        self.batched_iterator = BatchedTensorIterator(tensor=images_bhwc.movedim(-1, 1), sub_batch_size=batch_size,
                                                      device=self.target_device, dtype=self.target_dtype)

    def get_images(self, batch_range):
        images_bchw_pre = images_bchw = self.batched_iterator.get_batch(batch_range)
        if self.needs_scale:
            self.logger.debug(f"Scaling the input image from {images_bchw_pre.shape} to {self.model_w}x{self.model_h}")
            images_bchw = self.image_preproc.proc(images_bchw_pre)
        if not self.gen_outs:
            # We won't need them, so we can release the reference
            images_bchw_pre = None
        return images_bchw, images_bchw_pre

    def init_outs(self, image_compose):
        self.gen_outs = image_compose is not None
        if self.gen_outs:
            self.outs_bhwc_list = []

    def collect_outs(self, image_bchw):
        if not self.gen_outs:
            return
        self.outs_bhwc_list.append(self.scale_to_source(image_bchw).movedim(1, -1).to(device="cpu", dtype=self.out_dtype))

    def get_outs(self):
        if not self.gen_outs:
            return None
        outs_bhwc = torch.cat(self.outs_bhwc_list, dim=0)
        del self.outs_bhwc_list
        return outs_bhwc

    def run_single_inference(self, batch_range):
        images_bchw, images_bchw_pre = self.get_images(batch_range)
        if self.needs_map:
            masks_bchw = self.model(images_bchw, self.get_depths(batch_range, images_bchw))
        else:
            masks_bchw = self.model(images_bchw)
        # Optional threshold
        if self.mask_threshold > 0:
            masks_bchw = filter_mask(masks_bchw, threshold=self.mask_threshold)
        masks_bchw_scaled = self.collect_masks(masks_bchw)  # Keep the scaled copy if needed
        self.collect_edges()
        # The first two are only used to compose an output image, otherwise they are None
        return images_bchw_pre, masks_bchw_scaled, masks_bchw

    def run_inference(self, images_bhwc, depths_bhw, batch_size,
                      model_w=0, model_h=0, scale_method=DEFAULT_UPSCALE, preproc_img=False,  # Optional scale to model
                      mask_threshold=0.000,  # Optional mask threshold
                      image_compose=None,    # Optional image composition function
                      keep_depths=True, keep_edges=True, keep_masks=True, out_dtype=None):
        profiler = TorchProfile(self.logger, 2, f"profile for `{self.get_name()}` ({self.target_dtype})", self.target_device)

        self.init_images(images_bhwc, batch_size, preproc_img, model_w, model_h, scale_method, out_dtype)
        self.init_depths(depths_bhw, batch_size, keep_depths)
        self.init_masks(keep_masks, mask_threshold)
        self.init_edges(keep_edges)
        self.init_outs(image_compose)

        self.show_inference_info(images_bhwc, depths_bhw)

        with model_to_target(self.logger, self.model):
            with self.get_depths_context():
                for batch_range in self.batched_iterator:
                    images_bchw, masks_bchw_scaled, masks_bchw = self.run_single_inference(batch_range)
                    if image_compose is not None:
                        # Here both, image and mask, are on the model device and with its dtype
                        self.collect_outs(image_compose.compose(images_bchw, masks_bchw_scaled, batch_range))
                        del images_bchw
                        del masks_bchw_scaled
                    del masks_bchw
        outs, masks, depths, edges = (self.get_outs(), self.get_masks(), self.get_all_depths(), self.get_edges())

        profiler.end()

        return outs, masks, depths, edges
