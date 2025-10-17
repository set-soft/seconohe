# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# Helper to iterate a batch using batch_size slices
# Original code from Gemini 2.5 Pro
# Usage patterns:
# ------------------ Simple
# batched_iterator = BatchedTensorIterator(
#     tensor=large_batch_cpu,
#     sub_batch_size=SUB_BATCH_SIZE,
#     device=TARGET_DEVICE
# )
# for i, batch_range in enumerate(batched_iterator):
#     sub_batch = batched_iterator.get_batch(batch_range)
#
#     # Use the slice here
#
#     del sub_batch  # Avoid two instances at the same time in the target device
# ------------------ Process in separated function
# - Here we don't need exmplicit `del`
# for i, batch_range in enumerate(batched_iterator):
#     do_something(batched_iterator.get_batch(batch_range))
# ------------------
# Or in some cases, when we could exit the loop prematurelly
# for i, batch_range in enumerate(batched_iterator):
#     sub_batch = None  # Ensure variable exists before the try block
#     try:
#         # 1. Acquire the resource
#         sub_batch = batched_iterator.get_batch(batch_range)
#
#         # 2. Use the resource
#         # Your processing code goes here.
#         # If an error happens, the `finally` block is still executed.
#
#     finally:
#         # 3. Guarantee the resource is released
#         if sub_batch is not None:
#             del sub_batch
# ------------------
# The `del` at the end ensures we don't have 2 sub-batches in the target device at once.
# Can be omitted for small slices.
#
import math
import torch
from tqdm import tqdm
from typing import Union, Optional

# ComfyUI imports
try:
    import comfy.utils
    with_comfy = True
except ImportError:
    with_comfy = False


class BatchedTensorIterator:
    """
    An iterator that yields index ranges for sub-batches, allowing for
    manual control over tensor creation and memory management.

    This pattern avoids memory spikes by separating iteration from data loading.
    The user gets a `range` object in the loop and then calls `get_batch()`
    to create the tensor slice on the target device when they are ready.
    """
    def __init__(self, tensor: torch.Tensor, sub_batch_size: int, device: Union[str, torch.device],
                 dtype: Optional[torch.dtype] = None, show_progress: Optional[bool] = True,
                 progress_desc: Optional[str] = ""):
        """
        Initializes the iterator.

        Args:
            tensor (torch.Tensor): The large tensor to iterate over.
            sub_batch_size (int): The maximum size of each sub-batch.
            device (torch.device or str): The target device for tensor creation.
            dtype (torch.dtype, optional): The data type in the target device.
            show_progress (bool, optional): Show progress, default: True
            progress_desc (str, optional): Text to describe the progress.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input 'tensor' must be a torch.Tensor.")
        if not sub_batch_size > 0:
            raise ValueError("sub_batch_size must be a positive integer.")

        self.tensor = tensor
        self.sub_batch_size = sub_batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        self.total_size = tensor.shape[0]
        self.current_index = 0
        self.show_progress = show_progress
        self.elements = 0 if self.total_size == 0 else math.ceil(self.total_size / sub_batch_size)
        self.progress_desc = progress_desc
        self.pbar = None
        self.pbar_gui = None

    def __iter__(self):
        """Returns the iterator object itself."""
        self.current_index = 0
        # TQDM progress bar
        if self.pbar:
            self.pbar.close()
        if self.show_progress and self.elements > 1:
            self.pbar = tqdm(total=self.elements, desc=self.progress_desc)
            # ComfyUI progress bar
            if with_comfy:
                self.pbar_gui = comfy.utils.ProgressBar(self.elements)
        return self

    def __next__(self) -> range:
        """
        Calculates and returns the next index range for a sub-batch.

        Raises:
            StopIteration: When all ranges have been yielded.

        Returns:
            range: A lightweight range object (e.g., range(0, 8)).
        """
        if self.current_index >= self.total_size:
            if self.pbar:
                self.pbar.close()
                self.pbar = None
            self.pbar_gui = None
            raise StopIteration

        start = self.current_index
        end = min(start + self.sub_batch_size, self.total_size)
        self.current_index = end

        if self.pbar:
            self.pbar.update(1)
        if self.pbar_gui:
            self.pbar_gui.update(1)

        return range(start, end)

    def __len__(self) -> int:
        """Returns the total number of sub-batches that will be produced."""
        return self.elements

    def get_batch(self, batch_range: range) -> torch.Tensor:
        """
        Creates and returns a tensor slice for the given range, moving it
        to the target device, using the target dtype.

        Args:
            batch_range (range): The range of indices to slice from the main tensor.

        Returns:
            torch.Tensor: The resulting sub-batch tensor on the target device.
        """
        return self.tensor[batch_range.start:batch_range.stop].to(device=self.device, dtype=self.dtype)

    def get_aux_batch(self, aux_tensor: torch.Tensor, batch_range: range) -> torch.Tensor:
        """
        Creates and returns a tensor slice from an auxiliary tensor for the given range.

        If the auxiliary tensor is shorter than the requested range, its last element
        is repeated to fill the batch. This repetition is a memory-efficient view
        (using .expand()) and does not duplicate data in memory.

        Args:
            aux_tensor (torch.Tensor): The tensor from which to get the slice.
            batch_range (range): The range of indices to slice.

        Returns:
            torch.Tensor: The resulting sub-batch tensor on the target device.
        """
        # --- 1. Handle edge cases ---
        if aux_tensor is None:
            return None

        aux_len = aux_tensor.shape[0]
        if aux_len == 0:
            raise ValueError("Cannot get an auxiliary batch from an empty tensor.")

        start, end = batch_range.start, batch_range.stop
        requested_size = end - start

        # --- 2. The simple case: The auxiliary tensor is long enough ---
        if end <= aux_len:
            return aux_tensor[batch_range].to(device=self.device, dtype=self.dtype)

        # --- 3. The complex case: The auxiliary tensor is too short ---

        # Get the last available element from the auxiliary tensor.
        # We unsqueeze at dim=0 to give it a batch dimension of 1, e.g., (1, C, H, W).
        # This is a prerequisite for expanding along the batch dimension.
        last_element = aux_tensor[-1].unsqueeze(0)

        # --- 3a. No overlap: The requested range is entirely beyond the aux tensor ---
        if start >= aux_len:
            # We need to repeat the last element for the entire requested size.
            # .expand() creates a memory-efficient view of the tensor.
            expanded_part = last_element.expand(requested_size, *last_element.shape[1:])
            return expanded_part.to(device=self.device, dtype=self.dtype)

        # --- 3b. Partial overlap: The range starts within but ends outside ---
        else:
            # Get the part of the slice that actually exists.
            real_part = aux_tensor[start:]

            # Calculate how many repetitions of the last element are needed.
            num_repeats = end - aux_len

            # Create the memory-efficient view for the repeated part.
            expanded_part = last_element.expand(num_repeats, *last_element.shape[1:])

            # Combine the real slice and the expanded view into a single tensor.
            combined_tensor = torch.cat([real_part, expanded_part], dim=0)
            return combined_tensor.to(device=self.device, dtype=self.dtype)
