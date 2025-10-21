# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: SeCoNoHe
#
# String to color tuple conversion
import logging
import re
try:
    import PIL.ImageColor
    with_pil = True
except Exception:
    with_pil = False
COLORS_BY_NAME = {
    # Monochrome
    "black": (0.0, 0.0, 0.0),
    "white": (255.0, 255.0, 255.0),
    "gray": (128.0, 128.0, 128.0),
    "grey": (128.0, 128.0, 128.0),  # Common alternative spelling
    "silver": (192.0, 192.0, 192.0),

    # Pymatting green
    "green_ck": (120.0, 255.0, 155.0),

    # Primary (RGB)
    "red": (255.0, 0.0, 0.0),
    "green": (0.0, 255.0, 0.0),
    "blue": (0.0, 0.0, 255.0),

    # Secondary (CMY)
    "yellow": (255.0, 255.0, 0.0),
    "cyan": (0.0, 255.0, 255.0),
    "magenta": (255.0, 0.0, 255.0),

    # Other Common Colors
    "orange": (255.0, 165.0, 0.0),
    "purple": (128.0, 0.0, 128.0),
    "pink": (255.0, 192.0, 203.0),
    "brown": (165.0, 42.0, 42.0),
    "olive": (128.0, 128.0, 0.0),
    "teal": (0.0, 128.0, 128.0),
    "navy": (0.0, 0.0, 128.0),
}


def color_to_rgb(logger: logging.Logger, color_str: str) -> tuple[float, float, float]:
    """
    Parses a color string into a high-precision tuple of (R, G, B) floats in the [0, 255] range.
    This is the core parsing engine.
    It also supports using [0,1] values, 140 names, #RGB, #RRGGBB, rgb(R,G,B), rgb(r%,g%,b%),
    hsl(H,S%,L%) and hsv(H,S%,V%).

    :param logger: A logger instance.
    :param color_str: The string to convert.
    :return: A tuple with R, G, B components as floats, clamped to the 0-255 range.
    """
    s = str(color_str).strip().lower()

    # --- 1. Use PIL.ImageColor if available
    # Supports some 140 names, #RGB, #RRGGBB, rgb(R,G,B), rgb(r%,g%,b%), hsl(H,S%,L%), hsv(H,S%,V%)
    if with_pil:
        try:
            rgb = PIL.ImageColor.getrgb(s)[:3]
            return tuple(map(lambda x: float(x), rgb))
        except ValueError:
            pass

    # --- 2. A few colors by name
    if s in COLORS_BY_NAME:
        return COLORS_BY_NAME[s]

    # --- 3. Hexadecimal Parsing Block ---
    hex_s = s
    if hex_s.startswith('#'):
        hex_s = hex_s[1:]

    if len(hex_s) == 6 and re.match(r"^[0-9a-f]+$", hex_s):
        # Hex always directly converts to the 0-255 integer range.
        return tuple(float(int(hex_s[i:i+2], 16)) for i in (0, 2, 4))[0:3]

    # --- 4. Component Parsing Block ---
    try:
        parts = [float(p.strip()) for p in s.split(',')]
    except (ValueError, TypeError):
        logger.warning(f"Unrecognized color format or invalid value: '{color_str}'. Defaulting to black.")
        return (0.0, 0.0, 0.0)

    if len(parts) == 1:
        parts = parts * 3
    elif len(parts) == 2:
        parts.append(0.0)
    elif len(parts) > 3:
        logger.warning(f"Too many components in color string: '{color_str}'. Using first three.")
        parts = parts[:3]

    # Intelligently determine if the range is [0, 1] or [0, 255]
    if any(p > 1.0 for p in parts):
        # Assume [0, 255] range.
        return tuple(max(0.0, min(255.0, p)) for p in parts)
    else:
        # Assume [0, 1] range, scale to [0, 255].
        return tuple(max(0.0, min(255.0, p * 255.0)) for p in parts)


def color_to_rgb_uint8(logger: logging.Logger, color: str) -> tuple[int, int, int]:
    """
    Wrapper to get a color tuple of uint8 (0-255).
    Handles rounding correctly.
    """
    # Get the high-precision intermediate floats
    res_float_255 = color_to_rgb(logger, color)
    # Perform the rounding and casting to int here
    res_uint8 = tuple(int(round(p + 1e-6)) for p in res_float_255)
    logger.debug(f"Color '{color}' -> uint8 {res_uint8}")
    return res_uint8


def color_to_rgb_float(logger: logging.Logger, color: str) -> tuple[float, float, float]:
    """
    Wrapper to get a color tuple of float (0.0-1.0).
    Avoids intermediate rounding to preserve precision.
    """
    # Get the high-precision intermediate floats
    res_float_255 = color_to_rgb(logger, color)
    # Scale to the [0, 1] range without loss of precision
    res_float = tuple(c / 255.0 for c in res_float_255)
    logger.debug(f"Color '{color}' -> float {res_float}")
    return res_float


# ==============================================================================
# SECTION: Unit Tests
# ==============================================================================

def run_tests():
    """Runs a comprehensive suite of tests for the final color parsing functions."""
    logging.basicConfig(level=logging.INFO)
    test_logger = logging.getLogger("ColorTest")

    # The ground truth is the float value in the [0, 1] range.
    test_cases = [
        # --- Description, Input String, Expected Float (0-1 range) ---
        ("Hex with #", "#FF8000", (1.0, 128/255.0, 0.0)),
        ("Hex without #", "00ff80", (0.0, 1.0, 128/255.0)),
        ("Hex lowercase", "#abcdef", (171/255.0, 205/255.0, 239/255.0)),
        ("Hex with spaces", "  #0000FF  ", (0.0, 0.0, 1.0)),
        ("Invalid Hex", "#12345G", (0.0, 0.0, 0.0)),
        ("Short Hex 1", "#123", (17/255.0, 34/255.0, 51/255.0)),
        ("Short Hex 2", "#12", (0.0, 0.0, 0.0)),

        ("Int Components 1", "255, 128, 0", (1.0, 128/255.0, 0.0)),
        ("Int Components 2", "rgb(255,128,0)", (1.0, 128/255.0, 0.0)),
        ("Int Components with spaces", " 10,  20 ,30 ", (10/255.0, 20/255.0, 30/255.0)),
        ("Float Components", "1.0, 0.5, 0.0", (1.0, 0.5, 0.0)),
        ("Float Components [0,1] no spaces", "0.1,0.2,0.3", (0.1, 0.2, 0.3)),

        ("Grayscale Int", "128", (128/255.0, 128/255.0, 128/255.0)),
        ("Grayscale Float", "0.5", (0.5, 0.5, 0.5)),
        ("Two Components Int", "255, 128", (1.0, 128/255.0, 0.0)),
        ("Two Components Float", "1.0, 0.25", (1.0, 0.25, 0.0)),

        ("Clamping High", "300, 128, 500", (1.0, 128/255.0, 1.0)),
        ("Clamping Low", "-50, 128, -10", (0.0, 128/255.0, 0.0)),

        ("Mixed Int/Float > 1", "128, 0.5", (128/255.0, 0.5/255.0, 0.0)),  # Correctly parsed as 0-255 range
        ("Unsafe Float Heuristic 1", "255., 128, 0", (1.0, 128/255.0, 0.0)),

        ("Color name", "blue", (0.0, 0.0, 1.0)),

        ("Invalid Hex", "#12345G", (0.0, 0.0, 0.0)),
        ("Invalid Text", "glue", (0.0, 0.0, 0.0)),
        ("Invalid components", "255, 128, abc", (0.0, 0.0, 0.0)),
    ]

    passed_count = 0
    failed_count = 0

    print("--- Running Final Color Conversion Tests ---")
    for description, color_str, expected_float in test_cases:
        # Derive the expected uint8 from the float ground truth using the same logic as the function
        expected_uint8 = tuple(int(round(p * 255.0 + 1e-6)) for p in expected_float)

        print(f"\nTesting: {description} ('{color_str}')")

        # Test uint8 function
        try:
            res_uint8 = color_to_rgb_uint8(test_logger, color_str)
            assert res_uint8 == expected_uint8, f"Uint8 test failed! Expected {expected_uint8}, got {res_uint8}"
            print("  ✅ Uint8 PASSED")
            passed_count += 1
        except AssertionError as e:
            print(f"  ❌ Uint8 FAILED: {e}")
            failed_count += 1

        # Test float function
        try:
            res_float = color_to_rgb_float(test_logger, color_str)
            assert np.allclose(res_float, expected_float, atol=1e-6), \
                f"Float test failed! Expected {expected_float}, got {res_float}"
            print("  ✅ Float PASSED")
            passed_count += 1
        except AssertionError as e:
            print(f"  ❌ Float FAILED: {e}")
            failed_count += 1

    print("\n" + "="*40)
    print("--- Test Summary ---")
    print(f"Total assertions: {len(test_cases) * 2}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print("="*40)


if __name__ == '__main__':
    import numpy as np
    run_tests()
