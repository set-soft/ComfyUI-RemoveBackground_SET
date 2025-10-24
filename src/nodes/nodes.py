# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-RemoveBackground_SET
import json
import os
import safetensors.torch
from safetensors import safe_open
from seconohe.downloader import download_file
from seconohe.apply_mask import apply_mask
from seconohe.torch import get_torch_device_options, get_canonical_device
from seconohe.bti import BatchedTensorIterator
# from seconohe.torch import get_pytorch_memory_usage_str
import torch
from comfy import model_management
import folder_paths
from . import (main_logger, MODELS_DIR_KEY, MODELS_DIR, BATCHED_OPS, DEFAULT_UPSCALE, CATEGORY_BASIC, CATEGORY_LOAD,
               CATEGORY_ADV)
from .utils.arch import RemBg
from .utils.inspyrenet_config import parse_inspyrenet_config


logger = main_logger
auto_device_type = model_management.get_torch_device().type
models_path_default = folder_paths.get_folder_paths(MODELS_DIR_KEY)[0]
#
# Common options and choices
#
TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
WIDTH_OPT = ("INT", {
                "default": 1024,
                "min": 0,
                "max": 16384,
                "step": 32,
                "tooltip": "The width of the pre-processing image, does not affect the final output image size"
                })
HEIGHT_OPT = ("INT", {
                "default": 1024,
                "min": 0,
                "max": 16384,
                "step": 32,
                "tooltip": "The height of the pre-processing image, does not affect the final output image size"
                })
UPSCALE_OPT = (["area", "bicubic", "nearest-exact", "bilinear", "lanczos"], {
                "default": DEFAULT_UPSCALE,
                "tooltip": "Interpolation method for pre-processing image and post-processing mask"
                })
BLUR_SIZE_OPT = ("INT", {"default": 90, "min": 1, "max": 255, "step": 1, })
BLUR_SIZE_TWO_OPT = ("INT", {"default": 6, "min": 1, "max": 255, "step": 1, })
COLOR_OPT = ("STRING", {
                "default": "#000000",
                "tooltip": "Color for fill.\n"
                           "Can be an hexadecimal (#RRGGBB).\n"
                           "Can comma separated RGB values in [0-255] or [0-1.0] range."})
MASK_THRESHOLD_OPT = ("FLOAT", {"default": 0.000, "min": 0.0, "max": 1.0, "step": 0.001, })
DTYPE_OPS = (["AUTO", "float32", "float16"], {"default": "AUTO"})
DEPTH_OPS = ("MASK", {"tooltip": "For models that starts with a depth map"})
DIFFDIS_VAE = ("VAE", {"tooltip": "SD Turbo VAE for DiffDIS"})
POSITIVE = ("CONDITIONING", {"tooltip": "Experimental for DiffDIS"})


def dtype_str_to_torch(dtype: str) -> torch.dtype:
    return None if dtype is None or dtype == "AUTO" else TORCH_DTYPE[dtype]


class ModelInfo:
    """
    A simple class to hold model information.
    Attributes are dynamically created from the keys of the input dictionary.
    """
    def __init__(self, data: dict):
        self.no_commercial = False
        # Iterate through the dictionary and set attributes on the instance
        for key, value in data.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the object."""
        # self.__dict__ holds all the attributes of the instance
        attributes = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ModelInfo({attributes})"


class KnownModelsLoader:
    """
    Loads model information from a JSON file and populates a dictionary
    with ModelInfo objects.
    """
    def __init__(self, filename="known_models.json"):
        # The dictionary to store model names -> ModelInfo objects
        self.models = {}

        # --- Safely locate the JSON file ---
        # __file__ is the path to the current script
        # os.path.dirname gets the directory of that script
        # os.path.join creates a platform-independent path to the file
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find {filename} in the script directory: {script_dir}")

        # --- Load and process the file ---
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Iterate through the top-level dictionary from the JSON
        for model_name, model_data in data.items():
            # Create a ModelInfo instance and store it
            self.models[model_name] = ModelInfo(model_data)


KNOWN_MODELS = KnownModelsLoader().models


def add_inspyrenet_models():
    """ Try to add the transparent-background Python module models """
    # Look for the config and return the models
    models = parse_inspyrenet_config(logger)
    model_t = 'InSPyReNet'
    for m in models:
        ops_name = f"{m.name} (from TB config)"
        name = f"{model_t} {ops_name}"
        if name in KNOWN_MODELS:
            # We already added it
            continue
        KNOWN_MODELS[name] = ModelInfo({'url': m.url, 'file_name': m.ckpt_name, 'train_w': m.base_size[0],
                                        'train_h': m.base_size[1], 'model_t': model_t, 'ops_name': ops_name,
                                        'name': m.name, 'file_t': 'pytorch', 'dtype': 'float32'})


add_inspyrenet_models()


class LoadModel:
    """ Load already downloaded model """
    @classmethod
    def INPUT_TYPES(cls):
        device_options, _ = get_torch_device_options(with_auto=True)
        return {
            "required": {
                "model": (folder_paths.get_filename_list(MODELS_DIR_KEY),),
                "device": (device_options, )
            },
            "optional": {
                "dtype": DTYPE_OPS,
                "vae": DIFFDIS_VAE,
                "positive": POSITIVE,
            }
        }

    RETURN_TYPES = ("SET_REMBG",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model_file"
    CATEGORY = CATEGORY_LOAD
    DESCRIPTION = ("Load background remove model from folder models/" + MODELS_DIR +
                   " or the path of birefnet configured in the extra YAML file")
    UNIQUE_NAME = "LoadRembgByBiRefNetModel_SET"
    DISPLAY_NAME = "Load RemBG model by file"

    def load_model_file(self, model, device, dtype="auto", vae=None, positive=None):
        model_path = model if os.path.isabs(model) else folder_paths.get_full_path(MODELS_DIR_KEY, model)

        # Load the state dict
        logger.debug(f"Loading model weights from {model_path}")
        if model_path.endswith(".safetensors"):
            # Try to get the metadata
            with safe_open(model_path, framework="pt", device="cpu") as f:
                loaded_metadata = f.metadata()
                if loaded_metadata:
                    for key, value in loaded_metadata.items():
                        logger.debug(f"  - {key}: {value}")
            # Load the weights
            state_dict = safetensors.torch.load_file(model_path, device="cpu")
        else:
            state_dict = torch.load(model_path, map_location="cpu")
            if 'model_state_dict' in state_dict:
                # BEN
                state_dict = state_dict['model_state_dict']

        # Check this is valid for a known model
        arch = RemBg(state_dict, logger, model, vae, positive)
        arch.check()
        target_device = get_canonical_device(auto_device_type if device == "AUTO" else device)
        logger.debug(f"Using {target_device} device")
        target_dtype = arch.dtype if dtype == "AUTO" else TORCH_DTYPE[dtype]
        logger.debug(f"Using {target_dtype} data type")

        # Create an instance
        arch.instantiate_model(state_dict, target_device, target_dtype)
        return arch,


class AutoDownloadBiRefNetModel(LoadModel):
    """ Base class for all the auto-downloaders """
    model_type = 'BiRefNet'

    @classmethod
    def INPUT_TYPES(cls):
        device_options, _ = get_torch_device_options(with_auto=True)
        cls.known_models = {v.ops_name: v for k, v in KNOWN_MODELS.items() if v.model_t == cls.model_type}
        inputs = {
            "required": {
                "model_name": (list(cls.known_models.keys()),),
                "device": (device_options,)
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }
        if cls.model_type == 'DiffDIS':
            inputs["optional"]["vae"] = DIFFDIS_VAE
            inputs["optional"]["positive"] = POSITIVE
        return inputs

    RETURN_TYPES = ("SET_REMBG", "INT", "INT",     "NORM_PARAMS")
    RETURN_NAMES = ("model", "train_w", "train_h", "norm_params")
    FUNCTION = "load_model"

    @classmethod
    def fill_description(cls):
        cls.DESCRIPTION = f"Auto download {cls.model_type} model to models/{MODELS_DIR}"
        cls.UNIQUE_NAME = cls.__name__ + "_SET"
        cls.DISPLAY_NAME = f"Load {cls.model_type} model by name"

    def load_model(self, model_name, device, dtype="float32", vae=None, positive=None):
        m = self.known_models[model_name]
        if m.file_name is None:
            # Use the name in the URL
            fname = os.path.basename(m.url)
        else:
            # Use the extension from the URL
            fname = m.file_name + os.path.splitext(m.url)[1]
        model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, fname)
        if model_full_path is None:
            download_file(logger, m.url, models_path_default, fname)
        res = super().load_model_file(fname, device, dtype, vae=vae, positive=positive)
        arch = res[0]
        # Known training sizes have priority over default architecture sizes
        arch.w = m.train_w
        arch.h = m.train_h
        arch.sub_type = m.name
        if m.no_commercial:
            logger.warning(f"`{arch.get_name()}` model isn't for commercial use!")
        return (arch, m.train_w, m.train_h, {"mean": arch.img_mean, "std": arch.img_std})


# BiRefNet is de default
AutoDownloadBiRefNetModel.fill_description()


class AutoDownloadBENModel(AutoDownloadBiRefNetModel):
    model_type = 'MVANet'


# People knows about BEN
AutoDownloadBENModel.model_type = 'MVANet/BEN'
AutoDownloadBENModel.fill_description()
AutoDownloadBENModel.model_type = 'MVANet'


class AutoDownloadInSPyReNetModel(AutoDownloadBiRefNetModel):
    model_type = 'InSPyReNet'


AutoDownloadInSPyReNetModel.fill_description()


class AutoDownloadU2NetModel(AutoDownloadBiRefNetModel):
    model_type = 'U-2-Net'


AutoDownloadU2NetModel.fill_description()


class AutoDownloadISNetModel(AutoDownloadBiRefNetModel):
    model_type = 'IS-Net'


AutoDownloadISNetModel.fill_description()


class AutoDownloadMODNetModel(AutoDownloadBiRefNetModel):
    model_type = 'MODNet'


AutoDownloadMODNetModel.fill_description()


class AutoDownloadPDFNetModel(AutoDownloadBiRefNetModel):
    model_type = 'PDFNet'


AutoDownloadPDFNetModel.fill_description()


class AutoDownloadDiffDISModel(AutoDownloadBiRefNetModel):
    model_type = 'DiffDIS'


AutoDownloadDiffDISModel.fill_description()


class GetMaskLow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
                "batch_size": BATCHED_OPS,
            },
            "optional": {
                "depths": DEPTH_OPS,
                "out_dtype": DTYPE_OPS,
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("masks", "depths", "edges")
    FUNCTION = "get_mask"
    CATEGORY = CATEGORY_ADV
    UNIQUE_NAME = "GetMaskLowByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask low level"

    def get_mask(self, model, images, batch_size, depths=None, out_dtype=None):
        return model.run_inference(images, depths, batch_size, keep_depths=True, keep_edges=True, keep_masks=True,
                                   out_dtype=dtype_str_to_torch(out_dtype))[1:]


class GetMask(GetMaskLow):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
                "width": WIDTH_OPT,
                "height": HEIGHT_OPT,
                "upscale_method": UPSCALE_OPT,
                "mask_threshold": MASK_THRESHOLD_OPT,
                "batch_size": BATCHED_OPS,
            },
            "optional": {
                "depths": DEPTH_OPS,
                "out_dtype": DTYPE_OPS,
            }
        }

    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "GetMaskByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask"

    def get_mask(self, model, images, width=1024, height=1024, upscale_method=DEFAULT_UPSCALE, mask_threshold=0.000,
                 batch_size=1, depths=None, out_dtype=None):
        return model.run_inference(images, depths, batch_size,
                                   model_w=width, model_h=height, scale_method=upscale_method, preproc_img=True,
                                   mask_threshold=mask_threshold,
                                   keep_depths=True, keep_edges=True, keep_masks=True,
                                   out_dtype=dtype_str_to_torch(out_dtype))[1:]


class Advanced(GetMask):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
                "width": WIDTH_OPT,
                "height": HEIGHT_OPT,
                "upscale_method": UPSCALE_OPT,
                "blur_size": BLUR_SIZE_OPT,
                "blur_size_two": BLUR_SIZE_TWO_OPT,
                "fill_color": ("BOOLEAN", {"default": False}),
                "color": COLOR_OPT,
                "mask_threshold": MASK_THRESHOLD_OPT,
                "batch_size": BATCHED_OPS,
            },
            "optional": {
                "depths": DEPTH_OPS,
                "background": ("IMAGE", {"tooltip": "Image to use as background"}),
                "out_dtype": DTYPE_OPS,
            }
        }

    RETURN_TYPES = ("IMAGE",  "MASK",  "MASK",  "MASK")
    RETURN_NAMES = ("images", "masks", "depths", "edges")
    FUNCTION = "rem_bg"
    CATEGORY = CATEGORY_ADV
    UNIQUE_NAME = "RembgByBiRefNetAdvanced_SET"
    DISPLAY_NAME = "Remove background (full)"

    def rem_bg(self, model, images, upscale_method=DEFAULT_UPSCALE, width=1024, height=1024, blur_size=91, blur_size_two=7,
               fill_color=False, color=None, mask_threshold=0.000, batch_size=True, depths=None, background=None,
               keep_misc=True, out_dtype=None):
        self.blur_size = blur_size
        self.blur_size_two = blur_size_two
        self.fill_color = fill_color
        self.color = color
        if background is not None:
            self.background_iterator = BatchedTensorIterator(tensor=background, sub_batch_size=batch_size,
                                                             device=model.target_device, dtype=model.target_dtype)
        else:
            self.background_iterator = None
        self.background = background
        return model.run_inference(images, depths, batch_size,
                                   model_w=width, model_h=height, scale_method=upscale_method, preproc_img=True,
                                   mask_threshold=mask_threshold,
                                   image_compose=self.apply_mask,
                                   keep_depths=keep_misc, keep_edges=keep_misc, keep_masks=keep_misc,
                                   out_dtype=dtype_str_to_torch(out_dtype))

    def apply_mask(self, images_bchw, masks_bchw, batch_range):
        background = (None if self.background_iterator is None else
                      self.background_iterator.get_aux_batch(self.background, batch_range))
        out_images = apply_mask(logger, images_bchw.movedim(1, -1), masks=masks_bchw.squeeze(1),
                                device=model_management.get_torch_device(),
                                blur_size=self.blur_size, blur_size_two=self.blur_size_two, fill_color=self.fill_color,
                                color=self.color, batched=True, background=background)
        return out_images.movedim(-1, 1)


class RemBGSimple(Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
                "batch_size": BATCHED_OPS,
            },
            "optional": {
                "depths": DEPTH_OPS,
                "background": ("IMAGE", {"tooltip": "Image to use as background"}),
                "out_dtype": DTYPE_OPS,
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "rem_bg"
    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "RembgByBiRefNet_SET"
    DISPLAY_NAME = "Remove background"

    def rem_bg(self, model, images, batch_size, depths=None, background=None, out_dtype=None):
        w = model.w
        h = model.h
        logger.debug(f"Using size {w}x{h}")
        return super().rem_bg(model, images, width=w, height=h, batch_size=batch_size, depths=depths, background=background,
                              keep_misc=False, out_dtype=out_dtype)[:1]
