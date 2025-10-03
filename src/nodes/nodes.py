# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-BiRefNet-SET
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
KNOWN_MODELS = {
    #
    # BiRefNet models
    #
    'General (424 MiB)': (  # FP16
        'https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors',
        'General', 1024, 1024, 'BiRefNet'),
    'General (HR=2048) (424 MiB)': (  # FP16
        'https://huggingface.co/ZhengPeng7/BiRefNet_HR/resolve/main/model.safetensors',
        'General-HR', 2048, 2048, 'BiRefNet'),
    'General Lite (170 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/main/model.safetensors',
        'General-Lite', 1024, 1024, 'BiRefNet'),
    'General 2K Lite (170 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet_lite-2K/resolve/main/model.safetensors',
        'General-Lite-2K', 2560, 1440, 'BiRefNet'),
    'General (LR=512) (424 MiB)': (  # FP16
        'https://huggingface.co/ZhengPeng7/BiRefNet_512x512/resolve/main/model.safetensors',
        'General-reso_512', 512, 512, 'BiRefNet'),
    'General legacy (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-legacy/resolve/main/model.safetensors',
        'General-legacy', 1024, 1024, 'BiRefNet'),
    # dynamic is from 256x256 to 2304x2304
    'General dynamic res (424 MiB)': (  # FP16
        'https://huggingface.co/ZhengPeng7/BiRefNet_dynamic/resolve/main/model.safetensors',
        'General-dynamic', 1024, 1024, 'BiRefNet'),
    # Portrait
    'Portrait (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-portrait/resolve/main/model.safetensors',
        'Portrait', 1024, 1024, 'BiRefNet'),
    # Matting (w/alpha)
    'Matting (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-matting/resolve/main/model.safetensors',
        'Matting', 1024, 1024, 'BiRefNet'),
    'Matting Lite (85 MiB)': (  # FP16
        'https://huggingface.co/ZhengPeng7/BiRefNet_lite-matting/resolve/main/model.safetensors',
        'Matting-Lite', 1024, 1024, 'BiRefNet'),
    'Matting (HR=2048) (424 MiB)': (  # FP16
        'https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting/resolve/main/model.safetensors',
        'Matting-HR', 2048, 2048, 'BiRefNet'),
    # Dichotomous Image Segmentation (DIS)
    'Dichotomous Img Seg (DIS) (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-DIS5K/resolve/main/model.safetensors',
        'DIS', 1024, 1024, 'BiRefNet'),
    'Dichotomous Img Seg (DIS) TR/TEs (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-DIS5K-TR_TEs/resolve/main/model.safetensors',
        'DIS-TR_TEs', 1024, 1024, 'BiRefNet'),
    # High-Resolution Salient Object Detection (HRSOD)
    'HR Salient Obj. Detect.(HRSOD) (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-HRSOD/resolve/main/model.safetensors',
        'HRSOD', 1024, 1024, 'BiRefNet'),
    # Camouflaged Object Detection (COD)
    'Camouflaged Obj. Detect.(COD) (844 MiB)': (
        'https://huggingface.co/ZhengPeng7/BiRefNet-COD/resolve/main/model.safetensors',
        'COD', 1024, 1024, 'BiRefNet'),
    'BRIA v2.0 (No Com! 844 MiB)': (
        'https://huggingface.co/1038lab/RMBG-2.0/resolve/main/model.safetensors',
        'BRIA-RMBG2_0', 1024, 1024, 'BiRefNet'),
    #
    # BEN models
    #
    'General BEN2 (363 MiB)': (
        'https://huggingface.co/PramaLLC/BEN2/resolve/main/model.safetensors',
        'BEN2_Base', 1024, 1024, 'BEN'),
    'General BEN (1.05 GiB)': (
        'https://huggingface.co/PramaLLC/BEN/resolve/main/BEN_Base.pth',
        None, 1024, 1024, 'BEN'),
    #
    # InSPyReNet models
    #
    'Base 1.2.12 (351 MiB)': (
        'https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base.pth',
        'InSPyReNet_1_2_12_base', 1024, 1024, 'InSPyReNet'),
    # Safetensors? https://huggingface.co/1038lab/inspyrenet/resolve/main/inspyrenet.safetensors
    'Fast 1.2.12 (351 MiB)': (
        'https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_fast.pth',
        'InSPyReNet_1_2_12_fast', 384, 384, 'InSPyReNet'),
    'Nightly 1.2.12 (351 MiB)': (
        'https://github.com/plemeri/transparent-background/releases/download/1.2.12/ckpt_base_nightly.pth',
        'InSPyReNet_1_2_12_base_nightly.pth', 1024, 1024, 'InSPyReNet'),
    #
    # U²-Net models
    #
    'Base (u2net 169 MiB)': (
        'https://huggingface.co/netradrishti/u2net-saliency/resolve/main/models/u2net.pth',
        None, 320, 320, 'U-2-Net'),
    'Small (u2netp 4.5 MiB)': (
        'https://huggingface.co/netradrishti/u2net-saliency/resolve/main/models/u2netp.pth',
        None, 320, 320, 'U-2-Net'),
    #
    # IS-Net models
    #
    'Base (isnet 169 MiB)': (
        'https://huggingface.co/Carve/isnet/resolve/main/isnet.pth',
        None, 1024, 1024, 'IS-Net'),
    'DIS5K (isnet-general-use 169 MiB)': (
        'https://huggingface.co/ClockZinc/IS-NET_pth/resolve/main/isnet-general-use.pth',
        None, 1024, 1024, 'IS-Net'),
    'CarveSet (isnet-97-carveset 169 MiB)': (
        'https://huggingface.co/Carve/isnet/resolve/main/isnet-97-carveset.pth',
        None, 1024, 1024, 'IS-Net'),
    'Anime (ISNet_anime-seg 195 MiB)': (
        'https://huggingface.co/skytnt/anime-seg/resolve/main/model.safetensors',
        'ISNet_anime-seg.safetensors', 640, 640, 'IS-Net'),
    'BRIA v1.4 (No Com! 169 MiB)': (
        'https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.safetensors',
        'BRIA-RMBG1_4.safetensors', 1024, 1024, 'IS-Net'),
    #
    # MODNet models
    #
    'Photo portrait (26 MiB)': (
        'https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt',
        None, 512, 512, 'MODNet'),
    'Webcam portrait (26 MiB)': (
        'https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_webcam_portrait_matting.ckpt',
        None, 512, 512, 'MODNet'),
    #
    # MVANet models
    #
    'Base (369 MiB MiB)': (
        'https://huggingface.co/creative-graphic-design/MVANet-checkpoints/resolve/main/Model_80.pth',
        None, 1024, 1024, 'MVANet'),
}
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


def add_inspyrenet_models():
    """ Try to add the transparent-background Python module models """
    # Look for the config and return the models
    models = parse_inspyrenet_config(logger)
    for m in models:
        name = f"{m.name} (from TB config)"
        if name in KNOWN_MODELS:
            # We already added it
            continue
        KNOWN_MODELS[name] = (m.url, m.ckpt_name, m.base_size[0], m.base_size[1], 'InSPyReNet')


add_inspyrenet_models()


def filter_mask(mask, threshold=4e-3):
    mask_binary = mask > threshold
    filtered_mask = mask * mask_binary
    return filtered_mask


class ImagePreprocessor:
    def __init__(self, arch, resolution, upscale_method) -> None:
        interpolation = transforms.InterpolationMode(upscale_method)
        self.transform_image = transforms.Compose([transforms.Resize(resolution, interpolation=interpolation),
                                                   transforms.Normalize(arch.img_mean, arch.img_std)])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


class LoadModel:
    """ Load already downloaded model """
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
    DESCRIPTION = ("Load background remove model from folder models/" + MODELS_DIR +
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
    """ Base class for all the auto-downloaders """
    model_type = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([k for k, v in KNOWN_MODELS.items() if v[4] == cls.model_type],),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": DTYPE_OPS
            }
        }

    RETURN_TYPES = ("SET_REMBG", "INT", "INT",)
    RETURN_NAMES = ("model", "train_w", "train_h", )
    FUNCTION = "load_model"

    @classmethod
    def fill_description(cls):
        cls.DESCRIPTION = f"Auto download {cls.model_type} model to models/{MODELS_DIR}"
        cls.UNIQUE_NAME = cls.__name__ + "_SET"
        cls.DISPLAY_NAME = f"Load {cls.model_type} model by name"

    def load_model(self, model_name, device, dtype="float32"):
        url, fname, w, h, _ = KNOWN_MODELS[model_name]
        if fname is None:
            # Use the name in the URL
            fname = os.path.basename(url)
        else:
            # Use the extension from the URL
            fname += os.path.splitext(url)[1]
        model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, fname)
        if model_full_path is None:
            download_file(logger, url, models_path_default, fname)
        res = super().load_model_file(fname, device, dtype)
        model, arch = res[0]
        arch.w = w
        arch.h = h
        return ((model, arch), w, h)


class AutoDownloadBiRefNetModel(AutoDownloadModel):
    model_type = 'BiRefNet'


AutoDownloadBiRefNetModel.fill_description()


class AutoDownloadBENModel(AutoDownloadModel):
    model_type = 'BEN'


AutoDownloadBENModel.fill_description()


class AutoDownloadInSPyReNetModel(AutoDownloadModel):
    model_type = 'InSPyReNet'


AutoDownloadInSPyReNetModel.fill_description()


class AutoDownloadU2NetModel(AutoDownloadModel):
    model_type = 'U-2-Net'


AutoDownloadU2NetModel.fill_description()


class AutoDownloadISNetModel(AutoDownloadModel):
    model_type = 'IS-Net'


AutoDownloadISNetModel.fill_description()


class AutoDownloadMODNetModel(AutoDownloadModel):
    model_type = 'MODNet'


AutoDownloadMODNetModel.fill_description()


class AutoDownloadMVANetModel(AutoDownloadModel):
    model_type = 'MVANet'


AutoDownloadMVANetModel.fill_description()


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
