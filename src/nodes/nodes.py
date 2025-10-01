import os
import safetensors.torch
from seconohe.downloader import download_file
from seconohe.apply_mask import apply_mask
# from seconohe.torch import get_pytorch_memory_usage_str
import torch
from torchvision import transforms
from comfy import model_management
import comfy.utils
import folder_paths
from . import main_logger, MODELS_DIR_KEY, MODELS_DIR
from .utils.arch import RemBgArch
from .utils.inspyrenet_config import parse_inspyrenet_config


logger = main_logger
auto_device_type = model_management.get_torch_device().type
models_path_default = folder_paths.get_folder_paths(MODELS_DIR_KEY)[0]
#
# BiRefNet models
#
USAGE_TO_WEIGHTS_FILE = {
    'General': ('BiRefNet', 'General', 1024, 1024),                                           # 444 MB
    'General (HR=2048)': ('BiRefNet_HR', 'General-HR', 2048, 2048),                           # 444 MB (FP16)
    'General Lite': ('BiRefNet_lite', 'General-Lite', 1024, 1024),                            # 178 MB
    'General 2K Lite': ('BiRefNet_lite-2K', 'General-Lite-2K', 2560, 1440),                   # 178 MB
    'General (LR=512)': ('BiRefNet_512x512', 'General-reso_512', 512, 512),                   # 444 MB (FP16)
    'General legacy': ('BiRefNet-legacy', 'General-legacy', 1024, 1024),                      # 885 MB
    # dynamic is from 256x256 to 2304x2304
    'General dynamic res': ('BiRefNet_dynamic', 'General-dynamic', 1024, 1024),               # 444 MB (FP16?)
    # Portrait
    'Portrait': ('BiRefNet-portrait', 'Portrait', 1024, 1024),                                # 885 MB
    # Matting (w/alpha)
    'Matting': ('BiRefNet-matting', 'Matting', 1024, 1024),                                   # 885 MB
    'Matting Lite': ('BiRefNet_lite-matting', 'Matting-Lite', 1024, 1024),                    # 89 MB (FP16?)
    'Matting (HR=2048)': ('BiRefNet_HR-matting', 'Matting-HR', 2048, 2048),                   # 444 MB (FP16)
    # Dichotomous Image Segmentation (DIS)
    'Dichotomous Img Seg (DIS)': ('BiRefNet-DIS5K', 'DIS', 1024, 1024),                       # 885 MB
    'Dichotomous Img Seg (DIS) TR/TEs': ('BiRefNet-DIS5K-TR_TEs', 'DIS-TR_TEs', 1024, 1024),  # 885 MB
    # High-Resolution Salient Object Detection (HRSOD)
    'HR Salient Obj. Detect.(HRSOD)': ('BiRefNet-HRSOD', 'HRSOD', 1024, 1024),                # 885 MB
    # Camouflaged Object Detection (COD)
    'Camouflaged Obj. Detect.(COD)': ('BiRefNet-COD', 'COD', 1024, 1024),                     # 885 MB
    # This model needs user and password
    # 'BRIA v2.0 (No Com! 844 MiB)': ('BRIA-RMBG2_0', 'https://huggingface.co/briaai/RMBG-2.0/resolve/main/model.safetensors',
    #                                 1024, 1024),
}
MODEL_NAME_LIST = list(USAGE_TO_WEIGHTS_FILE.keys())
#
# BEN models
#
USAGE_TO_WEIGHTS_FILE_BEN = {
    'General BEN2': ('BEN2', 'BEN2_Base', 1024, 1024),                                        # 381 MB
}
MODEL_NAME_LIST_BEN = list(USAGE_TO_WEIGHTS_FILE_BEN.keys())
#
# InSPyReNet models
#
USAGE_TO_WEIGHTS_FILE_INSPYRENET = {
    'Base 1.2.12 (351 MiB)': ('1.2.12/ckpt_base.pth', 'InSPyReNet_1_2_12_base.pth', 1024, 1024),  # 351 MiB
    'Fast 1.2.12 (351 MiB)': ('1.2.12/ckpt_fast.pth', 'InSPyReNet_1_2_12_fast.pth', 384, 384),    # 351 MiB
    'Nightly 1.2.12 (351 MiB)': ('1.2.12/ckpt_base_nightly.pth', 'InSPyReNet_1_2_12_base_nightly.pth', 1024, 1024),  # 351 MiB
}
MODEL_NAME_LIST_INSPYRENET = list(USAGE_TO_WEIGHTS_FILE_INSPYRENET.keys())
#
# U²-Net models
#
USAGE_TO_WEIGHTS_FILE_U2NET = {
    'Base (u2net 169 MiB)': ('https://huggingface.co/netradrishti/u2net-saliency/resolve/main/models', 'u2net.pth', 320, 320),
    'Small (u2netp 4.5 MiB)': ('https://huggingface.co/netradrishti/u2net-saliency/resolve/main/models', 'u2netp.pth', 320, 320),
    'BRIA 1.4 (No Com! 169 MiB)': ('https://huggingface.co/briaai/RMBG-1.4/resolve/main', 'BRIA-RMBG1_4.safetensors', 1024, 1024),
}
MODEL_NAME_LIST_U2NET = list(USAGE_TO_WEIGHTS_FILE_U2NET.keys())
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
DEFAULT_UPSCALE = transforms.InterpolationMode.BICUBIC.value
UPSCALE_OPT = ([mode.value for mode in transforms.InterpolationMode], {
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
BATCHED_OPS = ("BOOLEAN", {
                  "default": True,
                  "tooltip": ("Apply the masks at once.\n"
                              "Faster, needs more memory")})
CATEGORY_BASE = "RemBG_SET"
CATEGORY_BASIC = CATEGORY_BASE+"/Basic"
CATEGORY_LOAD = CATEGORY_BASE+"/Load"
CATEGORY_ADV = CATEGORY_BASE+"/Advanced"


def filter_mask(mask, threshold=4e-3):
    mask_binary = mask > threshold
    filtered_mask = mask * mask_binary
    return filtered_mask


def download_birefnet_model(model_name):
    """ Downloading model from huggingface. """
    models_dir = os.path.join(models_path_default)
    hf_dir, name = USAGE_TO_WEIGHTS_FILE[model_name][:2]
    if name.startswith('http'):
        # Special case, from a different repo
        url = name
        name = hf_dir
    else:
        url = f"https://huggingface.co/ZhengPeng7/{hf_dir}/resolve/main/model.safetensors"
    download_file(logger, url, models_dir, f"{name}.safetensors")


def download_ben_model(model_name):
    """ Downloading model from huggingface. """
    models_dir = os.path.join(models_path_default)
    hf_dir, name = USAGE_TO_WEIGHTS_FILE_BEN[model_name][:2]
    url = f"https://huggingface.co/PramaLLC/{hf_dir}/resolve/main/model.safetensors"
    download_file(logger, url, models_dir, f"{name}.safetensors")


def download_inspyrenet_model(model_name):
    """ Downloading model from GitHub. """
    models_dir = os.path.join(models_path_default)
    gh_release, name = USAGE_TO_WEIGHTS_FILE_INSPYRENET[model_name][:2]
    url = f"https://github.com/plemeri/transparent-background/releases/download/{gh_release}"
    download_file(logger, url, models_dir, f"{name}.pth")


class ImagePreprocessor:
    def __init__(self, arch, resolution, upscale_method) -> None:
        interpolation = transforms.InterpolationMode(upscale_method)
        self.transform_image = transforms.Compose([transforms.Resize(resolution, interpolation=interpolation),
                                                   transforms.Normalize(arch.img_mean, arch.img_std)])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


class LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(MODELS_DIR_KEY),),
                "device": (["AUTO", "CPU"], )
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }

    RETURN_TYPES = ("SET_REMBG",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model_file"
    CATEGORY = CATEGORY_LOAD
    DESCRIPTION = ("Load BiRefNet model from folder models/" + MODELS_DIR +
                   " or the path of birefnet configured in the extra YAML file")
    UNIQUE_NAME = "LoadRembgByBiRefNetModel_SET"
    DISPLAY_NAME = "Load RemBG model by file"

    def load_model_file(self, model, device, dtype="auto"):
        model_path = model if os.path.isabs(model) else folder_paths.get_full_path(MODELS_DIR_KEY, model)
        if device == "AUTO":
            device_type = auto_device_type
        else:
            device_type = "cpu"
        logger.debug(f"Using {device_type} device")

        # Load the state dict
        logger.debug(f"Loading model weights from {model_path}")
        if model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path, device=device_type)
        else:
            state_dict = torch.load(model_path, map_location=device_type)
            if 'model_state_dict' in state_dict:
                # BEN
                state_dict = state_dict['model_state_dict']

        # Check this is valid for a known model
        arch = RemBgArch(state_dict, logger, model)
        arch.check()
        dtype = arch.dtype if dtype == "AUTO" else TORCH_DTYPE[dtype]
        logger.debug(f"Using {dtype} data type")

        # Create an instance
        model = arch.instantiate_model()
        model.load_state_dict(state_dict)
        model.to(device_type, dtype=dtype)
        model.eval()
        return [(model, arch)]


class AutoDownloadModel(LoadModel):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_NAME_LIST,),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }

    RETURN_TYPES = ("SET_REMBG", "INT", "INT",)
    RETURN_NAMES = ("model", "train_w", "train_h", )
    FUNCTION = "load_model"
    DESCRIPTION = "Auto download BiRefNet model from huggingface to models/"+MODELS_DIR+"/{model_name}.safetensors"
    UNIQUE_NAME = "AutoDownloadBiRefNetModel_SET"
    DISPLAY_NAME = "Load BiRefNet model by name"

    def load_model(self, model_name, device, dtype="float32"):
        _, fname, w, h = USAGE_TO_WEIGHTS_FILE[model_name]
        model_file_name = f'{fname}.safetensors'
        model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, model_file_name)
        if model_full_path is None:
            download_birefnet_model(model_name)
        res = super().load_model_file(model_file_name, device, dtype)
        model, arch = res[0]
        arch.w = w
        arch.h = h
        return ((model, arch), w, h)


class AutoDownloadModelBEN(LoadModel):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_NAME_LIST_BEN,),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }

    RETURN_TYPES = ("SET_REMBG", "INT", "INT",)
    RETURN_NAMES = ("model", "train_w", "train_h", )
    FUNCTION = "load_model"
    DESCRIPTION = "Auto download BEN model from huggingface to models/"+MODELS_DIR+"/{model_name}.safetensors"
    UNIQUE_NAME = "AutoDownloadBENModel_SET"
    DISPLAY_NAME = "Load BEN model by name"

    def load_model(self, model_name, device, dtype="float32"):
        _, fname, w, h = USAGE_TO_WEIGHTS_FILE_BEN[model_name]
        model_file_name = f'{fname}.safetensors'
        model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, model_file_name)
        if model_full_path is None:
            download_ben_model(model_name)
        res = super().load_model_file(model_file_name, device, dtype)
        model, arch = res[0]
        arch.w = w
        arch.h = h
        return ((model, arch), w, h)


def add_inspyrenet_models():
    """ Try to add the transparent-background Python module models """
    # Look for the config and return the models
    models = parse_inspyrenet_config(logger)
    added = False
    for m in models:
        name = f"{m.name} (from TB config)"
        if name in USAGE_TO_WEIGHTS_FILE_INSPYRENET:
            # We already added it
            continue
        USAGE_TO_WEIGHTS_FILE_INSPYRENET[name] = (m.url, m.ckpt_name, m.base_size[0], m.base_size[1])
        added = True
    if added:
        global MODEL_NAME_LIST_INSPYRENET
        MODEL_NAME_LIST_INSPYRENET = list(USAGE_TO_WEIGHTS_FILE_INSPYRENET.keys())


class AutoDownloadModelInSPyReNet(LoadModel):
    @classmethod
    def INPUT_TYPES(cls):
        add_inspyrenet_models()
        return {
            "required": {
                "model_name": (MODEL_NAME_LIST_INSPYRENET,),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }

    RETURN_TYPES = ("SET_REMBG", "INT", "INT",)
    RETURN_NAMES = ("model", "train_w", "train_h", )
    FUNCTION = "load_model"
    DESCRIPTION = "Auto download InSPyReNet model from github to models/"+MODELS_DIR+"/{model_name}.pth"
    UNIQUE_NAME = "AutoDownloadInSPyReNetModel_SET"
    DISPLAY_NAME = "Load InSPyReNet model by name"

    def load_model(self, model_name, device, dtype="float32"):
        url, fname, w, h = USAGE_TO_WEIGHTS_FILE_INSPYRENET[model_name]
        if os.path.isabs(fname):
            # A model from the transparent-background config.yaml
            model_file_name = fname
            if not os.path.isfile(fname):
                download_file(logger, url, os.path.dirname(fname), os.path.basename(fname))
        else:
            # A model from our known list
            model_file_name = f'{fname}.pth'
            model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, model_file_name)
            if model_full_path is None:
                download_inspyrenet_model(model_name)
        res = super().load_model_file(model_file_name, device, dtype)
        model, arch = res[0]
        arch.w = w
        arch.h = h
        return ((model, arch), w, h)


class AutoDownloadModelU2Net(LoadModel):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_NAME_LIST_U2NET,),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }

    RETURN_TYPES = ("SET_REMBG", "INT", "INT",)
    RETURN_NAMES = ("model", "train_w", "train_h", )
    FUNCTION = "load_model"
    DESCRIPTION = "Auto download U²-Net model from huggingface to models/"+MODELS_DIR+"/{model_name}.pth"
    UNIQUE_NAME = "AutoDownloadU2NetModel_SET"
    DISPLAY_NAME = "Load U²-Net model by name"

    def load_model(self, model_name, device, dtype="float32"):
        url, fname, w, h = USAGE_TO_WEIGHTS_FILE_U2NET[model_name]
        model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, fname)
        if model_full_path is None:
            download_file(logger, url, models_path_default, fname)
        res = super().load_model_file(fname, device, dtype)
        res.append(w)
        res.append(h)
        return res


class GetMaskLow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
                "batched": BATCHED_OPS,
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    CATEGORY = CATEGORY_ADV
    UNIQUE_NAME = "GetMaskLowByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask low level"

    def get_mask(self, model, images, batched):
        model, _ = model
        one_torch = next(model.parameters())
        model_device = one_torch.device.type
        model_dtype = one_torch.dtype

        b, h, w, c = images.shape
        if h % 32 or w % 32:
            raise ValueError(f"Image size must be a multiple of 32 (not {w}x{h})")
        image_bchw = images.permute(0, 3, 1, 2)

        if batched:
            mask_bchw = model(image_bchw.to(model_device, dtype=model_dtype)).cpu().float()
        else:
            progress_bar_ui = comfy.utils.ProgressBar(b)
            _mask_bchw = []
            for each_image in image_bchw:
                with torch.no_grad():
                    each_mask = model(each_image.unsqueeze(0).to(model_device, dtype=model_dtype)).cpu().float()
                _mask_bchw.append(each_mask)
                progress_bar_ui.update(1)

            mask_bchw = torch.cat(_mask_bchw, dim=0)  # (b, 1, h, w)
            del _mask_bchw

        mask_bhw = mask_bchw.squeeze(1)  # Discard the channels, which is 1 and we get (b, h, w)
        return mask_bhw,


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
                "batched": BATCHED_OPS,
            }
        }

    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "GetMaskByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask"

    def get_mask(self, model, images, width=1024, height=1024, upscale_method=DEFAULT_UPSCALE, mask_threshold=0.000,
                 batched=True):
        _, arch = model
        b, h, w, c = images.shape
        image_bchw = images.permute(0, 3, 1, 2)

        image_preproc = ImagePreprocessor(arch, resolution=(height, width), upscale_method=upscale_method)
        im_tensor = image_preproc.proc(image_bchw)
        del image_preproc

        mask_bchw = super().get_mask(model, im_tensor.permute(0, 2, 3, 1), batched)[0].unsqueeze(1)

        # Back to the original size to match the image size
        mask = torch.nn.functional.interpolate(mask_bchw, size=(h, w), mode=upscale_method)

        # Optional thresold for the mask
        if mask_threshold > 0:
            mask = filter_mask(mask, threshold=mask_threshold)

        mask_bhw = mask.squeeze(1)  # Discard the channels, which is 1 and we get (b, h, w)
        return mask_bhw,


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
                "batched": BATCHED_OPS,
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = CATEGORY_ADV
    UNIQUE_NAME = "RembgByBiRefNetAdvanced_SET"
    DISPLAY_NAME = "Remove background (full)"

    def rem_bg(self, model, images, upscale_method=DEFAULT_UPSCALE, width=1024, height=1024, blur_size=91, blur_size_two=7,
               fill_color=False, color=None, mask_threshold=0.000, batched=True):

        masks = super().get_mask(model, images, width, height, upscale_method, mask_threshold, batched)

        logger.debug(f"Applying mask/s (batched={batched})")
        out_images = apply_mask(logger, images, masks=masks[0], device=model_management.get_torch_device(),
                                blur_size=blur_size, blur_size_two=blur_size_two, fill_color=fill_color, color=color,
                                batched=batched)

        return out_images, masks[0]


class BiRefNet(Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "RembgByBiRefNet_SET"
    DISPLAY_NAME = "Remove background"

    def rem_bg(self, model, images):
        w = model[1].w
        h = model[1].h
        logger.debug(f"Using size {w}x{h}")
        b = images.shape[0]
        return super().rem_bg(model, images, width=w, height=h, batched=b <= 8)
