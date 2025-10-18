# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# Apply a mask to an image using "Approximate Fast Foreground Colour Estimation"
# Most of the code is from https://github.com/lldacing/ComfyUI_BiRefNet_ll
import logging
import torch
import torch.nn.functional as F
from typing import Optional
from .foreground_estimation.affce import affce
from .color import color_to_rgb_float


def add_mask_as_alpha(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Add the (b, h, w) shaped mask as the 4th channel (alpha channel) of the (b, h, w, 3) shaped image.
    """
    # Check input shape
    assert image.dim() == 4 and image.size(-1) == 3, "The shape of image should be (b, h, w, 3)."
    assert mask.dim() == 3, "The shape of mask should be (b, h, w)"
    assert image.size(0) == mask.size(0) and image.size(1) == mask.size(1) and image.size(2) == mask.size(2), \
           "The batch, height, and width dimensions of the image and mask must be consistent"

    # Expand the mask to (b, h, w, 1)
    mask = mask[..., None]

    # Concatenate image and mask into (b, h, w, 4)
    image_with_alpha = torch.cat([image, mask], dim=-1)

    return image_with_alpha


def apply_mask(
    logger: logging.Logger,
    images: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    blur_size: int = 91,
    blur_size_two: int = 7,
    fill_color: bool = False,
    color: Optional[str] = None,
    batched: bool = True,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    b, h, w, c = images.shape
    if b != masks.shape[0]:
        raise ValueError("images and masks must have the same batch size")

    images_on_device = images.to(device)
    masks_on_device = masks.to(device)

    # Approximate Fast Foreground Colour Estimation
    _image_masked_tensor = affce(images_on_device, masks_on_device, r1=blur_size, r2=blur_size_two, batched=batched)
    del images_on_device

    if background is not None:
        mask_to_apply = masks_on_device.unsqueeze(3).expand_as(_image_masked_tensor)
        background = background.to(device)
        if background.shape != images.shape:
            # Make the background and foreground match its size
            background = F.interpolate(background.movedim(-1, 1), size=images.shape[-3:-1], mode='bicubic').movedim(1, -1)
        out_images = _image_masked_tensor * mask_to_apply + background.to(device) * (1 - mask_to_apply)
    elif fill_color and color is not None:
        color = color_to_rgb_float(logger, color)
        # (b, h, w, 3)
        background_color = torch.tensor(color, device=device, dtype=images.dtype).view(1, 1, 1, 3).expand(b, h, w, 3)
        # (b, h, w) => (b, h, w, 3)
        mask_to_apply = masks_on_device.unsqueeze(3).expand_as(_image_masked_tensor)
        out_images = _image_masked_tensor * mask_to_apply + background_color.to(device) * (1 - mask_to_apply)
        # (b, h, w, 3)=>(b, h, w, 3)
    else:
        # The non-mask corresponding parts of the image are set to transparent
        out_images = add_mask_as_alpha(_image_masked_tensor.cpu(), masks.cpu())

    return out_images
