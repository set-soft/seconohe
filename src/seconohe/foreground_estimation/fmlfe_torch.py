# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# PyTorch approximate implementation of "Fast Multi-Level Foreground Estimation"
#
# Most code by Gemini 2.5 Pro
# Slow and inaccurate, but doesn't need extra dependencies
#
import torch
import torch.nn.functional as F_torch
import numpy as np


# _resize_nearest_torch helper function remains the same.
def _resize_nearest_torch(src: torch.Tensor, h_dst: int, w_dst: int) -> torch.Tensor:
    # Use PyTorch's built-in interpolate for an efficient, vectorized resize
    is_multichannel = src.dim() == 3
    if is_multichannel:
        # unsqueeze(0) to add batch dim for interpolate
        return F_torch.interpolate(src.unsqueeze(0), size=(h_dst, w_dst), mode='nearest')[0]
    else:
        # Handle 2D tensors
        return F_torch.interpolate(src.unsqueeze(0).unsqueeze(0), size=(h_dst, w_dst), mode='nearest')[0, 0]


# ==============================================================================
# Vectorized Red-Black Gauss-Seidel
# ==============================================================================
def _ml_iteration_torch_vectorized_gauss_seidel(
    F: torch.Tensor,
    B: torch.Tensor,
    image: torch.Tensor,
    alpha: torch.Tensor,
    regularization: float,
    gradient_weight: float,
    red_mask: torch.Tensor,
    black_mask: torch.Tensor
):
    """
    A fully vectorized implementation that correctly replicates the Gauss-Seidel
    behavior using a two-stage Red-Black update. This is both fast and correct.
    """

    def _calculate_update(F_in, B_in):
        padding = (1, 1, 1, 1)
        mode = 'replicate'
        F_padded = F_torch.pad(F_in, padding, mode=mode)
        B_padded = F_torch.pad(B_in, padding, mode=mode)
        alpha_padded = F_torch.pad(alpha, padding, mode=mode)
        F_neighbors = torch.stack([F_padded[:, :, 1:-1, :-2],
                                   F_padded[:, :, 1:-1, 2:],
                                   F_padded[:, :, :-2, 1:-1],
                                   F_padded[:, :, 2:, 1:-1]], dim=-1)
        B_neighbors = torch.stack([B_padded[:, :, 1:-1, :-2],
                                   B_padded[:, :, 1:-1, 2:],
                                   B_padded[:, :, :-2, 1:-1],
                                   B_padded[:, :, 2:, 1:-1]], dim=-1)
        alpha_neighbors = torch.stack([alpha_padded[:, :, 1:-1, :-2],
                                       alpha_padded[:, :, 1:-1, 2:],
                                       alpha_padded[:, :, :-2, 1:-1],
                                       alpha_padded[:, :, 2:, 1:-1]], dim=-1).squeeze(1)
        gradient = torch.abs(alpha.squeeze(1).unsqueeze(-1) - alpha_neighbors)
        da_neighbors = regularization + gradient_weight * gradient
        a_sum = torch.sum(da_neighbors, dim=-1).unsqueeze(1)
        b0_sum = torch.sum(da_neighbors.unsqueeze(1) * F_neighbors, dim=-1)
        b1_sum = torch.sum(da_neighbors.unsqueeze(1) * B_neighbors, dim=-1)
        a0, a1 = alpha, 1.0 - alpha
        b0, b1 = a0 * image + b0_sum, a1 * image + b1_sum
        a00, a11, a01 = a0 * a0 + a_sum, a1 * a1 + a_sum, a0 * a1
        inv_det = 1.0 / (a00 * a11 - a01 * a01 + 1e-8)
        F_new = torch.clamp(inv_det * (a11 * b0 - a01 * b1), 0.0, 1.0)
        B_new = torch.clamp(inv_det * (a00 * b1 - a01 * b0), 0.0, 1.0)
        return F_new, B_new

    # --- Stage 1: Update RED pixels, reading from original F and B ---
    F_red_update, B_red_update = _calculate_update(F, B)
    F[red_mask] = F_red_update[red_mask]
    B[red_mask] = B_red_update[red_mask]

    # --- Stage 2: Update BLACK pixels, reading from the NEWLY UPDATED RED neighbors in F and B ---
    F_black_update, B_black_update = _calculate_update(F, B)
    F[black_mask] = F_black_update[black_mask]
    B[black_mask] = B_black_update[black_mask]

    return F, B


def _estimate_fb_ml_torch(
    image: torch.Tensor,
    alpha: torch.Tensor,
    regularization: float,
    n_small_iterations: int,
    n_big_iterations: int,
    small_size: int,
    gradient_weight: float
):
    depth, h0, w0 = image.shape
    device = image.device

    # Compute average foreground and background color
    F_mask, B_mask = alpha > 0.9, alpha < 0.1
    F_count, B_count = F_mask.sum(), B_mask.sum()
    image_flat = image.view(depth, -1)
    F_mask_flat, B_mask_flat = F_mask.view(-1), B_mask.view(-1)
    F_mean_color = torch.sum(image_flat[:, F_mask_flat], dim=1) / (F_count + 1e-5)
    B_mean_color = torch.sum(image_flat[:, B_mask_flat], dim=1) / (B_count + 1e-5)

    # Initialize initial foreground and background with average color
    F_prev = F_mean_color.view(1, depth, 1, 1)
    B_prev = B_mean_color.view(1, depth, 1, 1)

    n_levels = int(np.ceil(np.log2(max(w0, h0))))

    for i_level in range(n_levels + 1):
        w = round(w0 ** (i_level / n_levels))
        h = round(h0 ** (i_level / n_levels))

        image_level = _resize_nearest_torch(image, h, w).unsqueeze(0)
        alpha_level = _resize_nearest_torch(alpha, h, w).unsqueeze(0).unsqueeze(0)
        F = _resize_nearest_torch(F_prev.squeeze(0), h, w).unsqueeze(0)
        B = _resize_nearest_torch(B_prev.squeeze(0), h, w).unsqueeze(0)

        # Create single-channel boolean masks first.
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        red_mask_1ch = ((y + x) % 2 == 0).unsqueeze(0).unsqueeze(0)
        black_mask_1ch = ~red_mask_1ch

        # Expand them to match the 3-channel tensors F and B.
        red_mask_expanded = red_mask_1ch.expand_as(F)
        black_mask_expanded = black_mask_1ch.expand_as(B)

        n_iter = n_small_iterations if min(w, h) <= small_size else n_big_iterations
        for _ in range(n_iter):
            F, B = _ml_iteration_torch_vectorized_gauss_seidel(
                F, B, image_level, alpha_level, regularization, gradient_weight,
                red_mask_expanded, black_mask_expanded
            )
        F_prev, B_prev = F, B

    return F.squeeze(0), B.squeeze(0)


def fmlfe_torch(
    image_tensor, alpha_tensor, regularization=1e-5, n_small_iterations=10,
    n_big_iterations=2, small_size=32, return_background=False, gradient_weight=1.0
):
    """
    Final PyTorch implementation that accepts single HWC tensors.
    """
    # The _estimate_fb_ml_torch function expects CHW, so we permute here.
    image_chw = image_tensor.permute(2, 0, 1)

    F_chw, B_chw = _estimate_fb_ml_torch(
        image_chw.to(torch.float32), alpha_tensor.to(torch.float32), float(regularization),
        int(n_small_iterations), int(n_big_iterations), int(small_size), float(gradient_weight)
    )

    # Permute back to HWC for a consistent output format
    F_out = F_chw.permute(1, 2, 0)
    B_out = B_chw.permute(1, 2, 0)

    return (F_out, B_out) if return_background else F_out
