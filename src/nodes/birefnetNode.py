import os
import safetensors.torch
import torch
from torchvision import transforms
from torch.hub import download_url_to_file
import comfy
from comfy import model_management
import folder_paths
from .birefnet.birefnet import BiRefNet
from .birefnet.birefnet_old import BiRefNet as OldBiRefNet
from .util import filter_mask, add_mask_as_alpha, refine_foreground_comfyui, check_state_dict
deviceType = model_management.get_torch_device().type

models_dir_key = "birefnet"

models_path_default = folder_paths.get_folder_paths(models_dir_key)[0]

usage_to_weights_file = {
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

modelNameList = list(usage_to_weights_file.keys())


def get_model_path(model_name):
    return os.path.join(models_path_default, f"{model_name}.safetensors")


def download_models(model_root, model_urls):
    if not os.path.exists(model_root):
        os.makedirs(model_root, exist_ok=True)

    for local_file, url in model_urls:
        local_path = os.path.join(model_root, local_file)
        if not os.path.exists(local_path):
            local_path = os.path.abspath(os.path.join(model_root, local_file))
            download_url_to_file(url, dst=local_path)


def download_birefnet_model(model_name):
    """
    Downloading model from huggingface.
    """
    model_root = os.path.join(models_path_default)
    model_urls = (
        (f"{model_name}.safetensors",
         f"https://huggingface.co/ZhengPeng7/{usage_to_weights_file[model_name]}/resolve/main/model.safetensors"),
    )
    download_models(model_root, model_urls)


interpolation_modes_mapping = {
    "nearest": 0,
    "bilinear": 2,
    "bicubic": 3,
    "nearest-exact": 0,
    # "lanczos": 1, #不支持
}


class ImagePreprocessor:
    def __init__(self, resolution, upscale_method="bilinear") -> None:
        interpolation = interpolation_modes_mapping.get(upscale_method, 2)
        self.transform_image = transforms.Compose([
            transforms.Resize(resolution, interpolation=interpolation),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform_image_old = transforms.Compose([
            transforms.Resize(resolution, interpolation=interpolation),
            # transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
        ])

    def proc(self, image) -> torch.Tensor:
        image = self.transform_image(image)
        return image

    def old_proc(self, image) -> torch.Tensor:
        image = self.transform_image_old(image)
        return image


VERSION = ["old", "v1"]
old_models_name = ["BiRefNet-DIS_ep580.pth", "BiRefNet-ep480.pth"]
torch_dtype = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class AutoDownloadBiRefNetModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (modelNameList,),
                "device": (["AUTO", "CPU"],)
            },
            "optional": {
                "dtype": (["float32", "float16"], {"default": "float32"})
            }
        }

    RETURN_TYPES = ("BIREFNET",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "image/BiRefNet"
    DESCRIPTION = "Auto download BiRefNet model from huggingface to models/BiRefNet/{model_name}.safetensors"

    def load_model(self, model_name, device, dtype="float32"):
        small = model_name in ["General-Lite", "General-Lite-2K", "Matting-Lite"]
        biRefNet_model = BiRefNet(small=small)
        model_file_name = f'{model_name}.safetensors'
        model_full_path = folder_paths.get_full_path(models_dir_key, model_file_name)
        if model_full_path is None:
            download_birefnet_model(model_name)
            model_full_path = folder_paths.get_full_path(models_dir_key, model_file_name)
        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"
        state_dict = safetensors.torch.load_file(model_full_path, device=device_type)
        biRefNet_model.load_state_dict(state_dict)
        biRefNet_model.to(device_type, dtype=torch_dtype[dtype])
        biRefNet_model.eval()
        return [(biRefNet_model, VERSION[1])]


class LoadRembgByBiRefNetModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (folder_paths.get_filename_list(models_dir_key),),
                "device": (["AUTO", "CPU"], )
            },
            "optional": {
                "use_weight": ("BOOLEAN", {"default": False}),
                "dtype": (["float32", "float16"], {"default": "float32"})
            }
        }

    RETURN_TYPES = ("BIREFNET",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "rembg/BiRefNet"
    DESCRIPTION = "Load BiRefNet model from folder models/BiRefNet or the path of birefnet configured in the extra YAML file"

    def load_model(self, model, device, use_weight=False, dtype="float32"):
        if model in old_models_name:
            version = VERSION[0]
            biRefNet_model = OldBiRefNet()
        else:
            version = VERSION[1]
            small = model in ["General-Lite", "General-Lite-2K", "Matting-Lite"]
            biRefNet_model = BiRefNet(small=small)

        model_path = folder_paths.get_full_path(models_dir_key, model)
        if device == "AUTO":
            device_type = deviceType
        else:
            device_type = "cpu"
        if model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path, device=device_type)
        else:
            state_dict = torch.load(model_path, map_location=device_type)
            state_dict = check_state_dict(state_dict)

        biRefNet_model.load_state_dict(state_dict)
        biRefNet_model.to(device_type, dtype=torch_dtype[dtype])
        biRefNet_model.eval()
        return [(biRefNet_model, version)]


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

    def get_mask(self, model, images, width=1024, height=1024, upscale_method='bilinear', mask_threshold=0.000):
        model, version = model
        one_torch = next(model.parameters())
        model_device_type = one_torch.device.type
        model_dtype = one_torch.dtype
        b, h, w, c = images.shape
        image_bchw = images.permute(0, 3, 1, 2)

        image_preproc = ImagePreprocessor(resolution=(height, width), upscale_method=upscale_method)
        if VERSION[0] == version:
            im_tensor = image_preproc.old_proc(image_bchw)
        else:
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
        # 遮罩大小需还原为与原图一致
        mask = comfy.utils.common_upscale(mask_bchw, w, h, upscale_method, "disabled")
        # (b, 1, h, w)
        if mask_threshold > 0:
            mask = filter_mask(mask, threshold=mask_threshold)
        # else:
        #   似乎几乎无影响
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
            # image的非mask对应部分设为透明 => (b, h, w, 4)
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

    def rem_bg(self, model, images):
        return super().rem_bg(model, images)


NODE_CLASS_MAPPINGS = {
    "AutoDownloadBiRefNetModel": AutoDownloadBiRefNetModel,
    "LoadRembgByBiRefNetModel": LoadRembgByBiRefNetModel,
    "RembgByBiRefNet": RembgByBiRefNet,
    "RembgByBiRefNetAdvanced": RembgByBiRefNetAdvanced,
    "GetMaskByBiRefNet": GetMaskByBiRefNet,
    "BlurFusionForegroundEstimation": BlurFusionForegroundEstimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoDownloadBiRefNetModel": "AutoDownloadBiRefNetModel",
    "LoadRembgByBiRefNetModel": "LoadRembgByBiRefNetModel",
    "RembgByBiRefNet": "RembgByBiRefNet",
    "RembgByBiRefNetAdvanced": "RembgByBiRefNetAdvanced",
    "GetMaskByBiRefNet": "GetMaskByBiRefNet",
    "BlurFusionForegroundEstimation": "BlurFusionForegroundEstimation",
}
