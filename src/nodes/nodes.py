import os
import safetensors.torch
from seconohe.downloader import download_file
import torch
from torchvision import transforms
import comfy
from comfy import model_management
import folder_paths
from . import main_logger, MODELS_DIR_KEY
from .util import filter_mask, add_mask_as_alpha, refine_foreground_comfyui
from .utils.arch import BiRefNetArch


logger = main_logger
auto_device_type = model_management.get_torch_device().type
models_path_default = folder_paths.get_folder_paths(MODELS_DIR_KEY)[0]
USAGE_TO_WEIGHTS_FILE = {
    'General': 'BiRefNet',
    'General-HR': 'BiRefNet_HR',
    'Matting-HR': 'BiRefNet_HR-matting',
    'General-Lite': 'BiRefNet_lite',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    'General-reso_512': 'BiRefNet_512x512',
    'Portrait': 'BiRefNet-portrait',
    'Matting': 'BiRefNet-matting',
    'Matting-Lite': 'BiRefNet_lite-matting',
    # 'Anime-Lite': 'BiRefNet_lite-Anime',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy',
    'General-dynamic': 'BiRefNet_dynamic',
}
MODEL_NAME_LIST = list(USAGE_TO_WEIGHTS_FILE.keys())
INTERPOLATION_MODES_MAPPING = {
    "nearest": 0,
    "bilinear": 2,
    "bicubic": 3,
    "nearest-exact": 0,
    # "lanczos": 1, # Not supported
}
TORCH_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    # "bfloat16": torch.bfloat16,
}


def download_birefnet_model(model_name):
    """ Downloading model from huggingface. """
    models_dir = os.path.join(models_path_default)
    url = f"https://huggingface.co/ZhengPeng7/{USAGE_TO_WEIGHTS_FILE[model_name]}/resolve/main/model.safetensors"
    download_file(logger, url, models_dir, f"{model_name}.safetensors")


class ImagePreprocessor:
    def __init__(self, arch, resolution, upscale_method="bilinear") -> None:
        interpolation = INTERPOLATION_MODES_MAPPING.get(upscale_method, 2)
        self.transform_image = transforms.Compose([transforms.Resize(resolution, interpolation=interpolation),
                                                   transforms.Normalize(arch.img_mean, arch.img_std)])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image


class LoadRembgByBiRefNetModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(MODELS_DIR_KEY),),
                "device": (["AUTO", "CPU"], )
            },
            "optional": {
                "dtype": (["float32", "float16"], {"default": "float32"})
            }
        }

    RETURN_TYPES = ("BIREFNET",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model_file"
    CATEGORY = "rembg/BiRefNet"
    DESCRIPTION = "Load BiRefNet model from folder models/BiRefNet or the path of birefnet configured in the extra YAML file"
    UNIQUE_NAME = "LoadRembgByBiRefNetModel_SET"
    DISPLAY_NAME = "Load BiRefNet model by file"

    def load_model_file(self, model, device, dtype="float32"):
        model_path = folder_paths.get_full_path(MODELS_DIR_KEY, model)
        if device == "AUTO":
            device_type = auto_device_type
        else:
            device_type = "cpu"

        # Load the state dict
        logger.debug(f"Loading model weights from {model_path}")
        if model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path, device=device_type)
        else:
            state_dict = torch.load(model_path, map_location=device_type)

        # Check this is valid for a known model
        arch = BiRefNetArch(state_dict, logger)
        arch.check()

        # Create an instance
        biRefNet_model = arch.instantiate_model()
        biRefNet_model.load_state_dict(state_dict)
        biRefNet_model.to(device_type, dtype=TORCH_DTYPE[dtype])
        biRefNet_model.eval()
        return [(biRefNet_model, arch)]


class AutoDownloadBiRefNetModel(LoadRembgByBiRefNetModel):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (MODEL_NAME_LIST,),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": (["float32", "float16"], {"default": "float32"})
            }
        }

    FUNCTION = "load_model"
    DESCRIPTION = "Auto download BiRefNet model from huggingface to models/BiRefNet/{model_name}.safetensors"
    UNIQUE_NAME = "AutoDownloadBiRefNetModel_SET"
    DISPLAY_NAME = "Load BiRefNet model by name"

    def load_model(self, model_name, device, dtype="float32"):
        model_file_name = f'{model_name}.safetensors'
        model_full_path = folder_paths.get_full_path(MODELS_DIR_KEY, model_file_name)
        if model_full_path is None:
            download_birefnet_model(model_name)
        return super().load_model_file(model_file_name, device, dtype)


class GetMaskByBiRefNet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BIREFNET",),
                "images": ("IMAGE",),
                "width": ("INT",
                          {
                              "default": 1024,
                              "min": 0,
                              "max": 16384,
                              "tooltip": "The width of the pre-processing image, does not affect the final output image size"
                          }),
                "height": ("INT",
                           {
                               "default": 1024,
                               "min": 0,
                               "max": 16384,
                               "tooltip": "The height of the pre-processing image, does not affect the final output image size"
                           }),
                "upscale_method": (["bilinear", "nearest", "nearest-exact", "bicubic"],
                                   {
                                       "default": "bilinear",
                                       "tooltip": "Interpolation method for pre-processing image and post-processing mask"
                                   }),
                "mask_threshold": ("FLOAT", {"default": 0.000, "min": 0.0, "max": 1.0, "step": 0.004, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "get_mask"
    CATEGORY = "rembg/BiRefNet"
    UNIQUE_NAME = "GetMaskByBiRefNet_SET"
    DISPLAY_NAME = "Get background mask (BiRefNet)"

    def get_mask(self, model, images, width=1024, height=1024, upscale_method='bilinear', mask_threshold=0.000):
        model, arch = model
        one_torch = next(model.parameters())
        model_device_type = one_torch.device.type
        model_dtype = one_torch.dtype
        b, h, w, c = images.shape
        image_bchw = images.permute(0, 3, 1, 2)

        image_preproc = ImagePreprocessor(arch, resolution=(height, width), upscale_method=upscale_method)
        im_tensor = image_preproc.proc(image_bchw)
        del image_preproc

        _mask_bchw = []
        for each_image in im_tensor:
            with torch.no_grad():
                each_mask = model(each_image.unsqueeze(0).to(model_device_type, dtype=model_dtype))[-1].sigmoid().cpu().float()
            _mask_bchw.append(each_mask)
            del each_mask

        mask_bchw = torch.cat(_mask_bchw, dim=0)
        del _mask_bchw
        # Back to the original size to match the image size
        mask = comfy.utils.common_upscale(mask_bchw, w, h, upscale_method, "disabled")
        # (b, 1, h, w)
        if mask_threshold > 0:
            mask = filter_mask(mask, threshold=mask_threshold)
        # else:
        #   Seems to have no effect
        #     mask = normalize_mask(mask)

        return mask.squeeze(1),


class BlurFusionForegroundEstimation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "blur_size": ("INT", {"default": 90, "min": 1, "max": 255, "step": 1, }),
                "blur_size_two": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1, }),
                "fill_color": ("BOOLEAN", {"default": False}),
                "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "get_foreground"
    CATEGORY = "rembg/BiRefNet"
    DESCRIPTION = "Approximate Fast Foreground Colour Estimation. https://github.com/Photoroom/fast-foreground-estimation"
    UNIQUE_NAME = "BlurFusionForegroundEstimation_SET"
    DISPLAY_NAME = "Blur fusion foreground estimation"

    def get_foreground(self, images, masks, blur_size=91, blur_size_two=7, fill_color=False, color=None):
        b, h, w, c = images.shape
        if b != masks.shape[0]:
            raise ValueError("images and masks must have the same batch size")

        device = model_management.get_torch_device()
        images_on_device = images.to(device)
        masks_on_device = masks.to(device)

        _image_masked_tensor = refine_foreground_comfyui(images_on_device, masks_on_device)

        if fill_color and color is not None:
            r = torch.full([b, h, w, 1], ((color >> 16) & 0xFF) / 0xFF)
            g = torch.full([b, h, w, 1], ((color >> 8) & 0xFF) / 0xFF)
            b = torch.full([b, h, w, 1], (color & 0xFF) / 0xFF)
            # (b, h, w, 3)
            background_color = torch.cat((r, g, b), dim=-1)
            # (b, 1, h, w) => (b, h, w, 3)
            apply_mask = masks_on_device.unsqueeze(3).expand_as(_image_masked_tensor)
            out_images = _image_masked_tensor * apply_mask + background_color.to(device) * (1 - apply_mask)
            # (b, h, w, 3)=>(b, h, w, 3)
            del background_color, apply_mask
        else:
            # The non-mask corresponding parts of the image are set to transparent
            out_images = add_mask_as_alpha(_image_masked_tensor.cpu(), masks.cpu())

        del _image_masked_tensor

        return out_images.cpu(), masks.cpu()


class RembgByBiRefNetAdvanced(GetMaskByBiRefNet, BlurFusionForegroundEstimation):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BIREFNET",),
                "images": ("IMAGE",),
                "width": ("INT",
                          {
                              "default": 1024,
                              "min": 0,
                              "max": 16384,
                              "tooltip": "The width of the pre-processing image, does not affect the final output image size"
                          }),
                "height": ("INT",
                           {
                               "default": 1024,
                               "min": 0,
                               "max": 16384,
                               "tooltip": "The height of the pre-processing image, does not affect the final output image size"
                           }),
                "upscale_method": (["bilinear", "nearest", "nearest-exact", "bicubic"],
                                   {
                                       "default": "bilinear",
                                       "tooltip": "Interpolation method for pre-processing image and post-processing mask"
                                   }),
                "blur_size": ("INT", {"default": 90, "min": 1, "max": 255, "step": 1, }),
                "blur_size_two": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1, }),
                "fill_color": ("BOOLEAN", {"default": False}),
                "color": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFF, "step": 1, "display": "color"}),
                "mask_threshold": ("FLOAT", {"default": 0.000, "min": 0.0, "max": 1.0, "step": 0.001, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/BiRefNet"
    UNIQUE_NAME = "RembgByBiRefNetAdvanced_SET"
    DISPLAY_NAME = "Remove background (BiRefNet) (full)"

    def rem_bg(self, model, images, upscale_method='bilinear', width=1024, height=1024, blur_size=91, blur_size_two=7, fill_color=False, color=None, mask_threshold=0.000):

        masks = super().get_mask(model, images, width, height, upscale_method, mask_threshold)

        out_images, out_masks = super().get_foreground(images, masks=masks[0], blur_size=blur_size, blur_size_two=blur_size_two, fill_color=fill_color, color=color)

        return out_images, out_masks


class RembgByBiRefNet(RembgByBiRefNetAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BIREFNET",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "rem_bg"
    CATEGORY = "rembg/BiRefNet"
    UNIQUE_NAME = "RembgByBiRefNet_SET"
    DISPLAY_NAME = "Remove background (BiRefNet)"

    def rem_bg(self, model, images):
        return super().rem_bg(model, images)
