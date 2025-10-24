# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# Wrapper for the ComfyUI ops.
# This is basically to avoid initializing layers that will be the loaded from the weights and to play nice
# with ComfyUI memory management. Also cast dtypes.
import torch.nn as nn


try:
    import comfy.ops
    Linear = comfy.ops.manual_cast.Linear
    Conv1d = comfy.ops.manual_cast.Conv1d
    Conv2d = comfy.ops.manual_cast.Conv2d
    Conv3d = comfy.ops.manual_cast.Conv3d
    GroupNorm = comfy.ops.manual_cast.GroupNorm
    LayerNorm = comfy.ops.manual_cast.LayerNorm
    ConvTranspose2d = comfy.ops.manual_cast.ConvTranspose2d
    ConvTranspose1d = comfy.ops.manual_cast.ConvTranspose1d
    RMSNorm = comfy.ops.manual_cast.RMSNorm
    Embedding = comfy.ops.manual_cast.Embedding
except ImportError:
    Linear = nn.Linear
    Conv1d = nn.Conv1d
    Conv2d = nn.Conv2d
    Conv3d = nn.Conv3d
    GroupNorm = nn.GroupNorm
    LayerNorm = nn.LayerNorm
    ConvTranspose2d = nn.ConvTranspose2d
    ConvTranspose1d = nn.ConvTranspose1d
    RMSNorm = nn.RMSNorm
    Embedding = nn.Embedding
