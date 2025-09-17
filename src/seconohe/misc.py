# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPL-3.0
# Project: SeCoNoHe
# Miscellaneous functions


def format_bytes(num_bytes: int) -> str:
    """
    Converts a number of bytes into a human-readable string with appropriate units
    (B, KB, MB, GB, TB).

    Args:
        num_bytes (int): The number of bytes to format.

    Returns:
        str: A human-readable string representation of the bytes.
    """
    if num_bytes is None:
        return "N/A"
    if num_bytes == 0:
        return "0 B"

    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}

    power = 0
    value = float(num_bytes)

    # Divide by 1024 until the value is less than 1024, incrementing the power
    while value >= 1024 and power < len(power_labels) - 1:
        value /= 1024
        power += 1

    return f"{value:.2f} {power_labels[power]}"
