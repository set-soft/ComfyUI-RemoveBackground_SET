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
from comfy.utils import ProgressBar, load_torch_file
import folder_paths

# Local imports
from . import main_logger as logger
from .depth_anything.dpt_v2 import DepthAnythingV2

BASE_URL = 'https://huggingface.co/Kijai/DepthAnythingV2-safetensors/resolve/main/'
KNOWN_MODELS = {
    #
    # BiRefNet models
    #
    'Small F16 (47 MiB)': 'depth_anything_v2_vits_fp16.safetensors',
    'Small F32 (95 MiB)': 'depth_anything_v2_vits_fp32.safetensors',
    'Base F16 (186 MiB)': 'depth_anything_v2_vitb_fp16.safetensors',
    'Base F32 (372 MiB)': 'depth_anything_v2_vitb_fp32.safetensors',
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
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "DepthAnythingV2"
    DESCRIPTION = "Create a depth map of the image\nSee: https://depth-anything-v2.github.io"
    UNIQUE_NAME = "DepthAnything_V2_SET"
    DISPLAY_NAME = "Depth Anything V2"

    def process(self, da_model, images):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = da_model['model']
        dtype = da_model['dtype']

        B, H, W, C = images.shape

        # images = images.to(device)
        images = images.permute(0, 3, 1, 2)

        orig_H, orig_W = H, W
        if W % 14 != 0:
            W = W - (W % 14)
        if H % 14 != 0:
            H = H - (H % 14)
        if orig_H % 14 != 0 or orig_W % 14 != 0:
            images = F.interpolate(images, size=(H, W), mode="bilinear")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalized_images = normalize(images)
        pbar = ProgressBar(B)
        out = []
        model.to(device)
        autocast_condition = (dtype != torch.float32) and not mm.is_device_mps(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            for img in normalized_images:
                depth = model(img.unsqueeze(0).to(device))
                depth = (depth - depth.min()) / (depth.max() - depth.min())
                out.append(depth.cpu())
                pbar.update(1)
            model.to(offload_device)
            depth_out = torch.cat(out, dim=0)
            depth_out = depth_out.unsqueeze(-1).repeat(1, 1, 1, 3).cpu().float()

        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if depth_out.shape[1] != final_H or depth_out.shape[2] != final_W:
            depth_out = F.interpolate(depth_out.permute(0, 3, 1, 2), size=(final_H, final_W),
                                      mode="bilinear").permute(0, 2, 3, 1)
        depth_out = (depth_out - depth_out.min()) / (depth_out.max() - depth_out.min())
        depth_out = torch.clamp(depth_out, 0, 1)
        if da_model['is_metric']:
            depth_out = 1 - depth_out
        return (depth_out,)
