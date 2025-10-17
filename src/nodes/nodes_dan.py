#
# Adapted from: https://github.com/kijai/ComfyUI-DepthAnythingV2
# Credits go to Kijai (Jukka Seppänen) https://github.com/kijai
#
# Adapted by Salvador E. Tropea
# Why?
# - Don't like the silent downloader (no progress and cryptic names)
# - Extra dependencies we can avoid, IDK why for "accelerate" if the code explicitly makes it optional
#
from contextlib import nullcontext
import os
from seconohe.downloader import download_file
from seconohe.bti import BatchedTensorIterator
from seconohe.torch import model_to_target
from seconohe.logger import get_debug_level
import torch
import torch.nn.functional as F
from torchvision import transforms
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except ImportError:
    pass

# ComfyUI imports
import comfy.model_management as mm
from comfy.utils import load_torch_file
import folder_paths

# Local imports
from . import main_logger as logger, BATCHED_OPS
from .depth_anything.dpt_v2 import DepthAnythingV2


BASE_URL = 'https://huggingface.co/Kijai/DepthAnythingV2-safetensors/resolve/main/'
BASE_MODEL_NAME = 'Base F32 (372 MiB)'
IMAGE_MOD = 14
KNOWN_MODELS = {
    #
    # BiRefNet models
    #
    'Small F16 (47 MiB)': 'depth_anything_v2_vits_fp16.safetensors',
    'Small F32 (95 MiB)': 'depth_anything_v2_vits_fp32.safetensors',
    'Base F16 (186 MiB)': 'depth_anything_v2_vitb_fp16.safetensors',
    BASE_MODEL_NAME: 'depth_anything_v2_vitb_fp32.safetensors',
    'Large F16 (640 MiB)': 'depth_anything_v2_vitl_fp16.safetensors',
    'Large F32 (1.3 GiB)': 'depth_anything_v2_vitl_fp32.safetensors',
    'Large Metric Indoor F32 (1.3 GiB)': 'depth_anything_v2_metric_hypersim_vitl_fp32.safetensors',
    'Large Metric Outdoor F32 (1.3 GiB)': 'depth_anything_v2_metric_vkitti_vitl_fp32.safetensors',
}


class DownloadAndLoadDepthAnythingV2Model:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (list(KNOWN_MODELS.keys()), ),
            },
        }

    RETURN_TYPES = ("DAMODEL",)
    RETURN_NAMES = ("da_v2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = ("Models autodownload to `ComfyUI/models/depthanything` from\n"
                   "https://huggingface.co/Kijai/DepthAnythingV2-safetensors/tree/main\n\n"
                   "F16 reduces quality by a LOT, not recommended.")
    UNIQUE_NAME = "DownloadAndLoadDepthAnythingV2Model_SET"
    DISPLAY_NAME = "Load Depth Anything by name"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        fname = KNOWN_MODELS[model]
        dtype = torch.float16 if "fp16" in fname else torch.float32
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            # 'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        custom_config = {
            'model_name': fname,
        }
        if not hasattr(self, 'model') or self.model is None or custom_config != self.current_config:
            self.current_config = custom_config
            download_path = os.path.join(folder_paths.models_dir, "depthanything")
            model_path = os.path.join(download_path, fname)

            if not os.path.exists(model_path):
                download_file(logger, BASE_URL+fname, download_path, fname)

            if "vitl" in fname:
                encoder = "vitl"
            elif "vitb" in fname:
                encoder = "vitb"
            elif "vits" in fname:
                encoder = "vits"

            if "hypersim" in fname:
                max_depth = 20.0
            else:
                max_depth = 80.0

            with (init_empty_weights() if is_accelerate_available else nullcontext()):
                if 'metric' in fname:
                    self.model = DepthAnythingV2(**{**model_configs[encoder], 'is_metric': True, 'max_depth': max_depth})
                else:
                    self.model = DepthAnythingV2(**model_configs[encoder])

            logger.debug(f"Loading Depth Anything V2 weights from {model_path}")
            state_dict = load_torch_file(model_path)
            if is_accelerate_available:
                for key in state_dict:
                    set_module_tensor_to_device(self.model, key, device=device, dtype=dtype, value=state_dict[key])
            else:
                self.model.load_state_dict(state_dict)

            self.model.eval()
            da_model = {
                "model": self.model,
                "dtype": dtype,
                "is_metric": self.model.is_metric
            }

        return (da_model,)


class DepthAnything_V2:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "da_model": ("DAMODEL", ),
            "images": ("IMAGE", ),
            "batch_size": BATCHED_OPS,
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("depths",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = "Create a depth map of the image\nSee: https://depth-anything-v2.github.io"
    UNIQUE_NAME = "DepthAnything_V2_SET"
    DISPLAY_NAME = "Depth Anything V2"

    def process(self, da_model, images, batch_size):
        model = da_model['model']
        model.target_dtype = da_model['dtype']
        model.target_device = mm.get_torch_device()
        with model_to_target(logger, model):
            depths_bchw = self.process_low(da_model, images.movedim(-1, 1), batch_size, out_dtype=images.dtype,
                                           out_device="cpu")
        return (depths_bchw.squeeze(1),)  # BHW

    def process_low(self, da_model, images_bchw, batch_size, out_dtype, out_device):
        model = da_model['model']
        is_metric = da_model['is_metric']
        B, C, H, W = images_bchw.shape

        # The model expects image sizes multiples than 14 (patch height)
        orig_H, orig_W = H, W
        if orig_H % IMAGE_MOD != 0 or orig_W % IMAGE_MOD != 0:
            needs_resize = True
            W = W - (W % IMAGE_MOD)
            H = H - (H % IMAGE_MOD)
        else:
            needs_resize = False

        debug_level = get_debug_level(logger)
        if debug_level >= 1:
            logger.debug(f"Starting Depth Anything V2 inference: {model.__class__.__name__}")
            if debug_level >= 2:
                logger.debug(f"- Model: {model.target_device}/{model.target_dtype} is_metric {is_metric}")
                logger.debug(f"- Input: {images_bchw.shape} {images_bchw.device}/{images_bchw.dtype} needs_resize: {needs_resize} ({W}x{H})")
                logger.debug(f"- Output: cpu/{out_dtype}")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        batched_iterator = BatchedTensorIterator(tensor=images_bchw, sub_batch_size=batch_size,
                                                 device=model.target_device, dtype=model.target_dtype)
        out = []
        for batch_range in batched_iterator:
            images_bchw = batched_iterator.get_batch(batch_range)
            # Pre-process
            if needs_resize:
                images_bchw = F.interpolate(images_bchw, size=(H, W), mode="bilinear")
            # Inference
            depth_bchw = model(normalize(images_bchw))
            del images_bchw
            # Post-process
            if needs_resize:
                depth_bchw = F.interpolate(depth_bchw, size=(orig_H, orig_W), mode="bilinear")
            depth_bchw = (depth_bchw - depth_bchw.min()) / (depth_bchw.max() - depth_bchw.min())  # Force [0,1]
            if is_metric:
                depth_bchw = 1 - depth_bchw
            out.append(depth_bchw.to(device=out_device, dtype=out_dtype))
            del depth_bchw

        return torch.cat(out, dim=0)
