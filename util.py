import numpy as np
import torch
from PIL import Image


def tensor_to_pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# ==============================================================================
# Implementation of cv2.blur using PyTorch (Gemini 2.5 Pro)
# ==============================================================================

def _pad_reflect_101(tensor, padding):
    """ A correct manual implementation of OpenCV's default BORDER_REFLECT_101 padding. """
    pad_left, pad_right, pad_top, pad_bottom = padding

    left_pad = tensor[:, :, :, 1:1 + pad_left].flip(dims=[3])
    right_pad = tensor[:, :, :, -1 - pad_right:-1].flip(dims=[3])
    tensor = torch.cat([left_pad, tensor, right_pad], dim=3)

    top_pad = tensor[:, :, 1:1 + pad_top, :].flip(dims=[2])
    bottom_pad = tensor[:, :, -1 - pad_bottom:-1, :].flip(dims=[2])
    tensor = torch.cat([top_pad, tensor, bottom_pad], dim=2)

    return tensor


def _blur_torch(np_array, kernel_size):
    """
    PyTorch replacement for the default cv2.blur().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_grayscale = np_array.ndim == 2
    if is_grayscale:
        tensor = torch.from_numpy(np_array).unsqueeze(0).unsqueeze(0)
    else:
        # Transpose from (H, W, C) to (N, C, H, W)
        tensor = torch.from_numpy(np_array.transpose((2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device, dtype=torch.float32)

    # Correct padding calculation for even/odd kernel anchor points
    pad_left_top = kernel_size // 2
    pad_right_bottom = (kernel_size - 1) - pad_left_top
    padding_tuple = (pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom)

    # Correct manual padding to match cv2.BORDER_REFLECT_101
    padded_tensor = _pad_reflect_101(tensor, padding_tuple)

    # Stable averaging operation
    pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=1)
    blurred_tensor = pool(padded_tensor)

    if is_grayscale:
        return blurred_tensor.squeeze().cpu().numpy()
    else:
        # Squeeze N dimension and transpose from (C, H, W) back to (H, W, C)
        return blurred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


### copied and modified image_proc.py
### This version adds r2 as parameter

def refine_foreground_pil(image, mask, r1=90, r2=6):
    if mask.size != image.size:
        mask = mask.resize(image.size)
    image = np.array(image) / 255.0
    mask = np.array(mask) / 255.0
    estimated_foreground = FB_blur_fusion_foreground_estimator_pil_2(image, mask, r1=r1, r2=r2)
    image_masked = Image.fromarray((estimated_foreground * 255.0).astype(np.uint8))
    return image_masked


def FB_blur_fusion_foreground_estimator_pil_2(image, alpha, r1=90, r2=6):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator_pil(
        image, image, image, alpha, r=r1)
    return FB_blur_fusion_foreground_estimator_pil(image, F, blur_B, alpha, r=r2)[0]


def FB_blur_fusion_foreground_estimator_pil(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = _blur_torch(alpha, (r, r))[:, :, None]

    blurred_FA = _blur_torch(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = _blur_torch(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B


def apply_mask_to_image(image, mask):
    """
    Apply a mask to an image and set non-masked parts to transparent.

    Args:
        image (torch.Tensor): Image tensor of shape (h, w, c) or (1, h, w, c).
        mask (torch.Tensor): Mask tensor of shape (1, 1, h, w) or (h, w).

    Returns:
        torch.Tensor: Masked image tensor of shape (h, w, c+1) with transparency.
    """
    # 判断 image 的形状
    if image.dim() == 3:
        pass
    elif image.dim() == 4:
        image = image.squeeze(0)
    else:
        raise ValueError("Image should be of shape (h, w, c) or (1, h, w, c).")

    h, w, c = image.shape
    # 判断 mask 的形状
    if mask.dim() == 4:
        mask = mask.squeeze(0).squeeze(0)  # 去掉前2个维度 (h,w)
    elif mask.dim() == 3:
        mask = mask.squeeze(0)
    elif mask.dim() == 2:
        pass
    else:
        raise ValueError("Mask should be of shape (1, 1, h, w) or (h, w).")

    assert mask.shape == (h, w), "Mask shape does not match image shape."

    # 将 mask 扩展到与 image 相同的通道数
    image_mask = mask.unsqueeze(-1).expand(h, w, c)

    # 应用遮罩，黑色部分是0，相乘后白色1的部分会被保留，其它部分变为了黑色
    masked_image = image * image_mask

    # 遮罩的黑白当做alpha通道的不透明度，黑色是0表示透明，白色是1表示不透明
    alpha = mask
    # alpha通道拼接到原图像的RGB中
    masked_image_with_alpha = torch.cat((masked_image[:, :, :3], alpha.unsqueeze(2)), dim=2)

    return masked_image_with_alpha.unsqueeze(0)


def normalize_mask(mask_tensor):
    max_val = torch.max(mask_tensor)
    min_val = torch.min(mask_tensor)

    if max_val == min_val:
        return mask_tensor

    normalized_mask = (mask_tensor - min_val) / (max_val - min_val)

    return normalized_mask

def add_mask_as_alpha(image, mask):
    """
    将 (b, h, w) 形状的 mask 添加为 (b, h, w, 3) 形状的 image 的第 4 个通道（alpha 通道）。
    """
    # 检查输入形状
    assert image.dim() == 4 and image.size(-1) == 3, "The shape of image should be (b, h, w, 3)."
    assert mask.dim() == 3, "The shape of mask should be (b, h, w)"
    assert image.size(0) == mask.size(0) and image.size(1) == mask.size(1) and image.size(2) == mask.size(2), "The batch, height, and width dimensions of the image and mask must be consistent"

    # 将 mask 扩展为 (b, h, w, 1)
    mask = mask[..., None]

    # 不做点乘，可能会有边缘轮廓线
    # image = image * mask
    # 将 image 和 mask 拼接为 (b, h, w, 4)
    image_with_alpha = torch.cat([image, mask], dim=-1)

    return image_with_alpha

def filter_mask(mask, threshold=4e-3):
    mask_binary = mask > threshold
    filtered_mask = mask * mask_binary
    return filtered_mask
