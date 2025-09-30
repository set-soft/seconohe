# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# Approximate Fast Foreground Colour Estimation
# https://ieeexplore.ieee.org/document/9506164
# DOI: 10.1109/ICIP42928.2021.9506164
# https://github.com/Photoroom/fast-foreground-estimation
# From BiRefNet implementation
# https://github.com/ZhengPeng7/BiRefNet/blob/main/image_proc.py
# Adapted to only use PyTorch (no need for OpenCV) using Gemini 2.5 Pro
import torch
import torchvision.transforms.functional as TF
from typing import Tuple


# ==============================================================================
# Implementation of cv2.blur using PyTorch (Gemini 2.5 Pro)
# ==============================================================================

def _pad_reflect_101(tensor: torch.Tensor, padding: Tuple[int, int, int, int]) -> torch.Tensor:
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


def _blur_torch(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
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


def _refine_foreground_tensor_batch(images: torch.Tensor, masks: torch.Tensor, r1: int = 90, r2: int = 6) -> torch.Tensor:
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


def _fb_blur_fusion_batch(
    image: torch.Tensor,
    F: torch.Tensor,
    B: torch.Tensor,
    alpha: torch.Tensor,
    r: int = 90
) -> Tuple[torch.Tensor, torch.Tensor]:
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


def affce(images_bhwc: torch.Tensor, masks_bhw: torch.Tensor, r1: int = 90, r2: int = 6, batched: bool = True) -> torch.Tensor:
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
    if batched:
        result_bchw = _refine_foreground_tensor_batch(images_bchw, masks_bchw, r1, r2)
    else:
        imgs = []
        for i in range(images_bchw.shape[0]):
            imgs.append(_refine_foreground_tensor_batch(images_bchw[i].unsqueeze(0), masks_bchw[i].unsqueeze(0), r1, r2))
        result_bchw = torch.cat(imgs, dim=0)

    # 3. Convert the final result from BCHW back to BHWC (once)
    result_bhwc = result_bchw.permute(0, 2, 3, 1)

    return result_bhwc
