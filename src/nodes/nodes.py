# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-BiRefNet-SET
import os
import safetensors.torch
from safetensors import safe_open
from seconohe.downloader import download_file
from seconohe.apply_mask import apply_mask
from seconohe.torch import get_torch_device_options, get_canonical_device
# from seconohe.torch import get_pytorch_memory_usage_str
import torch
from torchvision import transforms
from comfy import model_management
import folder_paths
from . import main_logger, MODELS_DIR_KEY, MODELS_DIR
from .utils.arch import RemBgArch
from .utils.inspyrenet_config import parse_inspyrenet_config
from .nodes_dan import DownloadAndLoadDepthAnythingV2Model, BASE_MODEL_NAME, DepthAnything_V2


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
    # MVANet/BEN models
    #
    'General BEN2 F16 (184 MiB)': (
        'https://huggingface.co/set-soft/RemBG/resolve/main/MVANet/BEN2_Base_F16.safetensors',
        None, 1024, 1024, 'MVANet'),
    'General BEN2 (363 MiB)': (
        'https://huggingface.co/PramaLLC/BEN2/resolve/main/model.safetensors',
        'BEN2_Base', 1024, 1024, 'MVANet'),
    'General BEN1 F16 (184 MiB)': (
        'https://huggingface.co/set-soft/RemBG/resolve/main/MVANet/BEN1_Base_F16.safetensors',
        None, 1024, 1024, 'MVANet'),
    'General BEN1 (1.05 GiB)': (
        'https://huggingface.co/PramaLLC/BEN/resolve/main/BEN_Base.pth',
        None, 1024, 1024, 'MVANet'),
    'MVANet F16 (184 MiB)': (
        'https://huggingface.co/set-soft/RemBG/resolve/main/MVANet/MVANet_80_F16.safetensors',
        None, 1024, 1024, 'MVANet'),
    'MVANet (369 MiB)': (
        'https://huggingface.co/creative-graphic-design/MVANet-checkpoints/resolve/main/Model_80.pth',
        None, 1024, 1024, 'MVANet'),
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
    # PDFNet models
    #
    'Base (392 MiB)': (
        'https://huggingface.co/Tennineee/PDFNet/resolve/main/PDFNet_Best.pth',
        None, 1024, 1024, 'PDFNet'),
    'General (392 MiB)': (
        'https://huggingface.co/Tennineee/PDFNet-General/resolve/main/PDF-Generally.pth',
        None, 1024, 1024, 'PDFNet'),
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
DEPTH_OPS = ("IMAGE", {"tooltip": "For models that starts with a depth map"})
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
    def __init__(self, img_mean, img_std, resolution, upscale_method) -> None:
        interpolation = transforms.InterpolationMode(upscale_method)
        self.transform_image = transforms.Compose([transforms.Resize(resolution, interpolation=interpolation),
                                                   # output[channel] = (input[channel] - mean[channel]) / std[channel]
                                                   transforms.Normalize(img_mean, img_std)])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


def scale_comfy_image(img, size, method):
    # BHWC -> BCHW -> interpolate -> BHWC
    return torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), size=size, mode=method).permute(0, 2, 3, 1)


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
        arch = RemBgArch(state_dict, logger, model)
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
        return {
            "required": {
                "model_name": ([k for k, v in KNOWN_MODELS.items() if v[4] == cls.model_type],),
                "device": (device_options,)
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
        arch = res[0]
        # Known training sizes have priority over default architecture sizes
        arch.w = w
        arch.h = h
        if "No Com!" in model_name:
            logger.warning("This particular model isn't for commercial use!")
        return (arch, w, h)


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


class GetMaskLow:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
                "batched": BATCHED_OPS,
            },
            "optional": {
                "depths": DEPTH_OPS,
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("masks", "depths")
    FUNCTION = "get_mask"
    CATEGORY = CATEGORY_ADV
    UNIQUE_NAME = "GetMaskLowByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask low level"

    def get_mask(self, model, images, batched, depths=None):
        # Check images and make them BCHW
        b, h, w, c = images.shape
        if h % 32 or w % 32:
            raise ValueError(f"Image size must be a multiple of 32 (not {w}x{h})")
        image_bchw = images.permute(0, 3, 1, 2)

        # Check depths and make them BCHW
        if model.needs_map:
            # PDFNet computes the mask using the image and a depth map
            if depths is None:
                # raise ValueError(f"For this model ({model.model_type}) you need to provide a depth map")
                depths = self.create_depth_maps(images)
            bm, hm, wm, cm = depths.shape
            if bm != b:
                raise ValueError(f"Found {b} images and {bm} depths, provide the same amount")
            if hm != h or wm != w:
                raise ValueError(f"Images using {w}x{h} and depths using {wm}x{hm}, must be of the same size")
            depth_bchw = depths.permute(0, 3, 1, 2)
        else:
            depth_bchw = torch.zeros((b, 3, 64, 64), dtype=torch.float32, device="cpu")

        return model.run_inference(image_bchw, depth_bchw, batched)

    def create_depth_maps(self, images):
        """ Automatically create depth maps using Depth Anything V2 vitb """
        # Do we have a model for this?
        if not hasattr(self, 'dan_model'):
            # Nope, create it
            self.dan_model = DownloadAndLoadDepthAnythingV2Model().loadmodel(BASE_MODEL_NAME)[0]
        return DepthAnything_V2().process(self.dan_model, images)[0]


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
            },
            "optional": {
                "depths": DEPTH_OPS,
            }
        }

    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "GetMaskByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask"

    def get_mask(self, model, images, width=1024, height=1024, upscale_method=DEFAULT_UPSCALE, mask_threshold=0.000,
                 batched=True, depths=None):
        arch = model
        b, h, w, c = images.shape
        image_bchw = images.permute(0, 3, 1, 2)

        image_preproc = ImagePreprocessor(arch.img_mean, arch.img_std, resolution=(height, width), upscale_method=upscale_method)
        im_tensor = image_preproc.proc(image_bchw)
        del image_preproc

        if depths is not None and arch.needs_map:
            depths_scaled = scale_comfy_image(depths, (height, width), upscale_method)
        else:
            depths_scaled = None

        mask_bhw, depths_bhwc = super().get_mask(model, im_tensor.permute(0, 2, 3, 1), batched, depths=depths_scaled)

        # Back to the original size to match the image size
        mask_bchw = torch.nn.functional.interpolate(mask_bhw.unsqueeze(1), size=(h, w), mode=upscale_method)

        # Optional thresold for the mask
        if mask_threshold > 0:
            mask_bchw = filter_mask(mask_bchw, threshold=mask_threshold)

        mask_bhw = mask_bchw.squeeze(1)  # Discard the channels, which is 1 and we get (b, h, w)

        # Depths
        if depths_scaled is None:
            # We created depth maps, real or empty
            depths = scale_comfy_image(depths_bhwc, (h, w), upscale_method)
        # Otherwise just pass the input depths
        return mask_bhw, depths


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
            },
            "optional": {
                "depths": DEPTH_OPS,
            }
        }

    RETURN_TYPES = ("IMAGE",  "MASK",  "IMAGE")
    RETURN_NAMES = ("images", "masks", "depths")
    FUNCTION = "rem_bg"
    CATEGORY = CATEGORY_ADV
    UNIQUE_NAME = "RembgByBiRefNetAdvanced_SET"
    DISPLAY_NAME = "Remove background (full)"

    def rem_bg(self, model, images, upscale_method=DEFAULT_UPSCALE, width=1024, height=1024, blur_size=91, blur_size_two=7,
               fill_color=False, color=None, mask_threshold=0.000, batched=True, depths=None):

        masks, depths = super().get_mask(model, images, width, height, upscale_method, mask_threshold, batched, depths=depths)

        logger.debug(f"Applying mask/s (batched={batched})")
        out_images = apply_mask(logger, images, masks=masks, device=model_management.get_torch_device(),
                                        blur_size=blur_size, blur_size_two=blur_size_two, fill_color=fill_color, color=color,
                                        batched=batched)

        return out_images, masks, depths


class RemBG(Advanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SET_REMBG",),
                "images": ("IMAGE",),
            },
            "optional": {
                "depths": DEPTH_OPS,
            }
        }

    FUNCTION = "rem_bg"
    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "RembgByBiRefNet_SET"
    DISPLAY_NAME = "Remove background"

    def rem_bg(self, model, images, depths=None):
        w = model.w
        h = model.h
        logger.debug(f"Using size {w}x{h}")
        b = images.shape[0]
        return super().rem_bg(model, images, width=w, height=h, batched=b <= 8, depths=depths)


from .diffdis.diffusers_local.src.diffusers import UNet2DConditionModel_diffdis
from .diffdis.diffdis_pipeline import DiffDISPipeline


class DiffDIS(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "positive": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("masks", "edges")
    FUNCTION = "diff_dis"
    CATEGORY = CATEGORY_BASIC
    UNIQUE_NAME = "DiffDIS_SET"
    DISPLAY_NAME = "DiffDIS"

    def diff_dis(self, images, positive, vae):
        # The model uses just 2 of the 77
        positive = positive[0][0][:, :2, :].to(auto_device_type)

        # The whole sd-turbo repo in models/sd-turbo
        pretrained_model_path = os.path.join(folder_paths.models_dir, "sd-turbo")
        # The DiffDIS trained checkpoint in models/diffdis/unet
        checkpoint_path = os.path.join(folder_paths.models_dir, "diffdis")

        # Build a DiffDIS pipeline
        unet = UNet2DConditionModel_diffdis.from_pretrained(
            checkpoint_path,
            subfolder="unet",
            in_channels=8,
            sample_size=96,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=False,
            class_embed_type='projection',
            projection_class_embeddings_input_dim=4,
            mid_extra_cross=True,
            mode='DBIA',
            use_swci=True)
        pipe = DiffDISPipeline(unet=unet, vae=vae)
        pipe = pipe.to(auto_device_type)

        # Pre-process the images
        b, h, w, c = images.shape
        image_bchw = images.permute(0, 3, 1, 2)

        # 1024x1024 and from [0,1] to [-1,1]
        image_preproc = ImagePreprocessor([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], resolution=(1024, 1024), upscale_method='bilinear')
        im_tensor = image_preproc.proc(image_bchw).to(auto_device_type)
        del image_preproc

        mask_bchw, edge_bchw = pipe(
            im_tensor,
            positive,
            ensemble_size=1,
            batch_size=1,
            show_progress_bar=False,
        )

        mask_bhw = torch.nn.functional.interpolate(mask_bchw, size=(h, w), mode='bilinear').squeeze(1).cpu()
        edge_bhw = torch.nn.functional.interpolate(edge_bchw, size=(h, w), mode='bilinear').squeeze(1).cpu()

        return mask_bhw, edge_bhw
