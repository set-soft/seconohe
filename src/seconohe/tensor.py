# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# Tensor manipulation helpers
import torch


def batched_min_max_norm(s: torch.Tensor, in_place: bool = True) -> torch.Tensor:
    """
    Performs per-sample min-max normalization on a batch of tensors.

    Each sample in the batch (along the 0-th dimension) is independently
    normalized to a [0, 1] range. This function is fully vectorized for
    maximum performance.

    Args:
        s (torch.Tensor): The input tensor, expected to have a shape of
                          (B, C, H, W) or similar batch format.
                          (B, H, W) and (B, H, W, C) ComfyUI images and
                          masks are also supported.
        in_place (bool, optional): If True, the input tensor `s` will be
                                   modified directly to save memory.
                                   If False, a clone of `s` is created and
                                   returned. Defaults to True.

    Returns:
        torch.Tensor: The normalized tensor. This will be the same object
                      as the input `s` if `in_place=True`.
    """
    if not in_place:
        s = s.clone()

    B = s.shape[0]

    # Flatten the spatial/channel dimensions to find min/max for each sample
    s_flat = s.view(B, -1)
    mi = torch.min(s_flat, dim=1, keepdim=True).values
    ma = torch.max(s_flat, dim=1, keepdim=True).values

    # Calculate the denominator, adding epsilon for stability
    denominator = ma - mi
    denominator[denominator == 0] = 1e-8  # Avoid division by zero for flat tensors

    # Perform the normalization directly on the flattened view.
    # Broadcasting handles the (B, N) shape with the (B, 1) min/max tensors.
    s_flat -= mi
    s_flat /= denominator

    # Because s_flat is a view of s, the original tensor s is now
    # normalized, with its original shape preserved. No reshaping needed.
    return s


def sigmoid_and_batched_min_max_norm(s: torch.Tensor, in_place: bool = True) -> torch.Tensor:
    """
    Applies a sigmoid function and then per-sample min-max normalization.

    This function combines sigmoid and normalization into a single utility,
    correctly handling in-place and out-of-place operations.

    Args:
        s (torch.Tensor): The input tensor.
        in_place (bool, optional): If True, the input tensor `s` will be
                                   modified directly. If False, a new tensor
                                   is created. Defaults to True.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    if in_place:
        # Apply sigmoid directly to the input tensor
        s.sigmoid_()
    else:
        # Create a new tensor for the sigmoid result
        s = torch.sigmoid(s)

    # Normalize the result. It's always safe to do this part in-place,
    # because if in_place=False, 's' is already a new tensor.
    return batched_min_max_norm(s, True)
