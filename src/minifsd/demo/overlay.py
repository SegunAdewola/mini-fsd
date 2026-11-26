"""Draws perception outputs onto a photo for the demo.

Each draw_* function takes an RGB image and one model output and returns a
new image, leaving the input untouched. The perception node (M4.6) uses
the same drawings for its RViz overlay.
"""

from collections.abc import Sequence

import cv2
import numpy as np

from minifsd.eval import detection

FLOOR_RGB = (0, 200, 80)  # green: "drivable"
FLOOR_ALPHA = 0.45  # enough tint to see the mask, enough photo to judge it
# One colour per detector class, none of them the floor's green.
BOX_RGB = {
    "person": (255, 60, 60),
    "obstacle": (255, 170, 0),
    "door": (60, 140, 255),
    "room_sign": (230, 230, 230),
}


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


def draw_boxes(
    rgb: np.ndarray, boxes: Sequence[detection.Detection]
) -> np.ndarray:
    """Outlines each box in its class's colour, labelled with its score.

    Args:
        rgb: (H, W, 3) uint8.
        boxes: Detections in this image's pixels.

    Returns:
        A new (H, W, 3) uint8 image.
    """
    out = np.ascontiguousarray(rgb.copy())
    for box in boxes:
        colour = BOX_RGB.get(box.category, (255, 255, 255))
        x0, y0 = round(box.x), round(box.y)
        x1, y1 = round(box.x + box.w), round(box.y + box.h)
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 1)
        cv2.putText(
            out,
            f"{box.category} {box.score:.2f}",
            (x0, max(y0 - 2, 8)),
            cv2.FONT_HERSHEY_PLAIN,
            0.7,
            colour,
            1,
        )
    return out
