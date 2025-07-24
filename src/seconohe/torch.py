# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# PyTorch helpers
import contextlib  # For context manager
import logging
import torch
from typing import Optional, Iterator, cast
try:
    import comfy.model_management as mm
    with_comfy = True
except Exception:
    with_comfy = False


def get_torch_device_options() -> tuple[list[str], str]:
    """
    Detects available PyTorch devices and returns a list and a suitable default.

    Scans for CPU, CUDA devices, and MPS (for Apple Silicon), providing a list
    of device strings (e.g., 'cpu', 'cuda', 'cuda:0') and a recommended default.

    :return: A tuple containing the list of available device strings and the
             recommended default device string.
    :rtype: tuple[list[str], str]
    """
    # We always have CPU
    default = "cpu"
    options = [default]
    # Do we have CUDA?
    if torch.cuda.is_available():
        default = "cuda"
        options.append(default)
        for i in range(torch.cuda.device_count()):
            options.append(f"cuda:{i}")  # Specific CUDA devices
    # Is this a Mac?
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        options.append("mps")
        if default == "cpu":
            default = "mps"
    return options, default


def get_offload_device() -> torch.device:
    """
    Gets the appropriate device for offloading models.

    Uses `comfy.model_management.unet_offload_device()` if available in a ComfyUI
    environment, otherwise defaults to the CPU.

    :return: The torch.device object to use for offloading.
    :rtype: torch.device
    """
    if with_comfy:
        return cast(torch.device, mm.unet_offload_device())
    return torch.device("cpu")


def get_canonical_device(device: str | torch.device) -> torch.device:
    """
    Converts a device string or object into a canonical torch.device object.

    Ensures that a device string like 'cuda' is converted to its fully
    indexed form, e.g., 'cuda:0', by checking the current default device.

    :param device: The device identifier to canonicalize.
    :type device: str | torch.device
    :return: A torch.device object with an explicit index if applicable.
    :rtype: torch.device
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # If it's a CUDA device and no index is specified, get the default one.
    if device.type == 'cuda' and device.index is None:
        # NOTE: This adds a dependency on torch.cuda.current_device()
        # The first solution is often better as it doesn't need this.
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    return device


# ##################################################################################
# # Helper for inference (Target device, offload, eval, no_grad and cuDNN Benchmark)
# ##################################################################################

@contextlib.contextmanager
def model_to_target(logger: logging.Logger, model: torch.nn.Module) -> Iterator[None]:
    """
    A context manager for safe model inference.

    Handles device placement, inference state (`eval()`, `no_grad()`),
    optional cuDNN benchmark settings, and safe offloading in a `finally` block
    to ensure resources are managed even if errors occur.

    The model object is expected to have two optional custom attributes:
    - ``target_device`` (torch.device): The device to run inference on.
    - ``cudnn_benchmark_setting`` (bool): The desired cuDNN benchmark state.

    Example Usage::

        with model_to_target(logger, my_model):
            # Your inference code here
            output = my_model(input_tensor)

    :param logger: A logger instance for logging state changes.
    :type logger: logging.Logger
    :param model: The torch.nn.Module to manage.
    :type model: torch.nn.Module
    :yield: None
    """
    if not isinstance(model, torch.nn.Module):
        with torch.no_grad():
            yield  # The code inside the 'with' statement runs here
        return

    # --- EXPLICIT TYPE ANNOTATIONS ---
    # Tell mypy the expected types for these variables.
    target_device: torch.device
    original_device: torch.device
    original_cudnn_benchmark_state: Optional[bool] = None  # Default is to keep the current setting

    # 1. Determine target device from the model object
    # Use a hasattr check for robustness, and type hinting
    if hasattr(model, 'target_device') and isinstance(model.target_device, torch.device):
        # Mypy now understands that model.target_device exists and is a torch.device
        target_device = model.target_device
    else:
        logger.warning("model_to_target: 'target_device' attribute not found or is not a torch.device on the model. "
                       "Defaulting to model's current device.")
        # Ensure model has parameters before calling next()
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            logger.warning("model_to_target: Model has no parameters. Cannot determine device. Assuming CPU.")
            target_device = torch.device("cpu")

    # 2. Get CUDNN benchmark setting from the model object (optional)
    # Use hasattr as this is an optional setting that not all models might have.
    cudnn_benchmark_enabled: Optional[bool] = None
    if hasattr(model, 'cudnn_benchmark_setting'):
        setting = model.cudnn_benchmark_setting
        if isinstance(setting, bool):
            cudnn_benchmark_enabled = setting
        else:
            logger.warning(f"model.cudnn_benchmark_setting was not a bool, but {type(setting)}. Ignoring.")

    try:
        # Get original_device from model after ensuring it has parameters
        original_device = next(model.parameters()).device
    except StopIteration:
        original_device = torch.device("cpu")  # Match fallback from above

    is_cuda_target = target_device.type == 'cuda'

    try:
        # 3. Manage cuDNN benchmark state
        if (cudnn_benchmark_enabled is not None and is_cuda_target and hasattr(torch.backends, 'cudnn') and
           torch.backends.cudnn.is_available()):
            if torch.backends.cudnn.benchmark != cudnn_benchmark_enabled:
                original_cudnn_benchmark_state = torch.backends.cudnn.benchmark
                torch.backends.cudnn.benchmark = cudnn_benchmark_enabled
                logger.debug(f"Temporarily set cuDNN benchmark to {torch.backends.cudnn.benchmark}")

        # 4. Move model to target device if not already there
        if original_device != target_device:
            logger.debug(f"Moving model from `{original_device}` to target device `{target_device}` for inference.")
            model.to(target_device)

        # 5. Set to eval mode and disable gradients for the operation
        model.eval()
        with torch.no_grad():
            yield  # The code inside the 'with' statement runs here

    finally:
        # 6. Restore original cuDNN benchmark state
        if original_cudnn_benchmark_state is not None:
            # This check is sufficient because it will only be not None if we set it inside the try block
            torch.backends.cudnn.benchmark = original_cudnn_benchmark_state
            logger.debug(f"Restored cuDNN benchmark to {original_cudnn_benchmark_state}")

        # 7. Offload model back to CPU
        if with_comfy:
            offload_device = get_offload_device()
            try:
                current_device_after_yield = next(model.parameters()).device
                if current_device_after_yield != offload_device:
                    logger.debug(f"Offloading model from `{current_device_after_yield}` to offload device `{offload_device}`.")
                    model.to(offload_device)
                    # Clear cache if we were on a CUDA device
                    if 'cuda' in str(current_device_after_yield):
                        torch.cuda.empty_cache()
            except StopIteration:
                pass  # Model with no parameters doesn't need offloading.
