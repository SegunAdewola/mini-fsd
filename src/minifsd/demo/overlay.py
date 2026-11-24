"""Draws perception outputs onto a photo for the demo.

Each draw_* function takes an RGB image and one model output and returns a
new image, leaving the input untouched. The floor is the only output for
now; depth, boxes and room numbers join it in M3 and M4.
"""

import numpy as np

FLOOR_RGB = (0, 200, 80)  # green: "drivable"
FLOOR_ALPHA = 0.45  # enough tint to see the mask, enough photo to judge it


def draw_floor(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tints the floor pixels green.

    Args:
        rgb: (H, W, 3) uint8.
        mask: (H, W) bool, True on the floor.

    Returns:
        A new (H, W, 3) uint8 image: floor pixels blended with FLOOR_RGB at
        FLOOR_ALPHA, all other pixels unchanged.

    Raises:
        ValueError: If mask and image sizes differ.
    """
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"mask {mask.shape} doesn't match image {rgb.shape}")
    out = rgb.copy()
    tint = np.array(FLOOR_RGB, np.float32)
    blended = (1 - FLOOR_ALPHA) * rgb[mask] + FLOOR_ALPHA * tint
    out[mask] = np.round(blended).astype(np.uint8)
    return out
