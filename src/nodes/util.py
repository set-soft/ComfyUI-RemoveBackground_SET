import torch
import torchvision.transforms.functional as TF


# ==============================================================================
# Implementation of cv2.blur using PyTorch (Gemini 2.5 Pro)
# ==============================================================================

def _pad_reflect_101(tensor, padding):
    """ A correct manual implementation of OpenCV's default BORDER_REFLECT_101 padding.
        Expects BCHW. """
    pad_left, pad_right, pad_top, pad_bottom = padding

    left_pad = tensor[:, :, :, 1:1 + pad_left].flip(dims=[3])
    right_pad = tensor[:, :, :, -1 - pad_right:-1].flip(dims=[3])
    tensor = torch.cat([left_pad, tensor, right_pad], dim=3)

    top_pad = tensor[:, :, 1:1 + pad_top, :].flip(dims=[2])
    bottom_pad = tensor[:, :, -1 - pad_bottom:-1, :].flip(dims=[2])
    tensor = torch.cat([top_pad, tensor, bottom_pad], dim=2)

    return tensor


def _blur_torch(tensor, kernel_size):
    """
    PyTorch replacement for the default cv2.blur().
    Input: Tensor of shape (B, C, H, W)
    Output: Tensor of shape (B, C, H, W)
    """
    # Correct padding calculation for even/odd kernel anchor points
    pad_left_top = kernel_size // 2
    pad_right_bottom = (kernel_size - 1) - pad_left_top
    padding_tuple = (pad_left_top, pad_right_bottom, pad_left_top, pad_right_bottom)

    # Correct manual padding to match cv2.BORDER_REFLECT_101
    padded_tensor = _pad_reflect_101(tensor, padding_tuple)

    # Stable averaging operation
    # AvgPool2d expects (N, C, H, W) and is channel-independent.
    pool = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=1)

    blurred_tensor = pool(padded_tensor)

    return blurred_tensor


def _refine_foreground_tensor_batch(images, masks, r1=90, r2=6):
    """
    A fully vectorized, tensor-based implementation of the foreground refinement.

    Args:
        images (torch.Tensor): Batch of images, shape (B, 3, H, W), float 0-1.
        masks (torch.Tensor): Batch of masks, shape (B, 1, H, W) or (B, 3, H, W), float 0-1.
    """
    # Ensure mask is single-channel (Luminance), equivalent to .convert("L")
    if masks.shape[1] == 3:
        masks = TF.rgb_to_grayscale(masks)  # Shape: (B, 1, H, W)

    # PyTorch's element-wise operations and broadcasting handle the dimensions.
    F, blur_B = _fb_blur_fusion_batch(images, images, images, masks, r=r1)
    estimated_foreground = _fb_blur_fusion_batch(images, F, blur_B, masks, r=r2)[0]

    return estimated_foreground


def _fb_blur_fusion_batch(image, F, B, alpha, r=90):
    """ Core algorithm, fully tensorized for batch processing. """
    # alpha is (B, 1, H, W), other tensors are (B, 3, H, W)

    # Blur the single-channel alpha mask
    blurred_alpha = _blur_torch(alpha, r)  # Output: (B, 1, H, W)

    # PyTorch broadcasting automatically handles `F * alpha`
    # (B, 3, H, W) * (B, 1, H, W) -> (B, 3, H, W)
    blurred_FA = _blur_torch(F * alpha, r)
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = _blur_torch(B * (1 - alpha), r)
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)

    # All tensors are now (B, 3, H, W)
    F_new = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F_new = torch.clamp(F_new, 0, 1)

    return F_new, blurred_B


def refine_foreground_comfyui(images_bhwc, masks_bhw, r1=90, r2=6):
    """
    The public-facing wrapper for ComfyUI.
    Handles the conversion from BHWC to BCHW and back.

    Args:
        images_bhwc (torch.Tensor): Batch of images, shape (B, H, W, 3), float 0-1.
        masks_bhwc (torch.Tensor): Batch of masks, shape (B, H, W), float 0-1.
    """
    # 1. Convert inputs from BHWC to BCHW (once)
    images_bchw = images_bhwc.permute(0, 3, 1, 2)
    masks_bchw = masks_bhw.unsqueeze(1)

    # 2. Call the internal engine which operates efficiently on BCHW
    result_bchw = _refine_foreground_tensor_batch(images_bchw, masks_bchw, r1, r2)

    # 3. Convert the final result from BCHW back to BHWC (once)
    result_bhwc = result_bchw.permute(0, 2, 3, 1)

    return result_bhwc


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


# TODO: Might be needed for some model, must be verified
def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict
