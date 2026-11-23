"""Classical floor segmentation: flood fill from the bottom centre.

The baseline that every learned floor model in E2 has to beat. It needs no
training, so it has no sim-to-real gap, and it runs in a few milliseconds.

How it works, in order:
    1. Blur slightly, so sensor noise doesn't stop the fill.
    2. Convert to LAB colour, where lightness (L) is separate from colour
       (a, b). Shading mostly changes L, so floor under uneven light stays
       close in a and b.
    3. Find edges with Canny and thicken them. Edges become barriers: the
       base of a wall or the outline of a box stops the fill.
    4. Flood fill from a seed near the bottom centre of the image, which is
       almost always floor for a forward-facing robot camera. Each pixel is
       compared with its already-filled neighbour, so the fill can follow a
       slow change in brightness across the floor.
    5. Close small holes (specks, floor-tile seams) with a morphological
       close.

It fails on textured floors (rugs, tiles with strong seams), reflections,
shadows, and whenever the bottom centre isn't floor.
"""

import dataclasses

import cv2
import numpy as np

FILL_VALUE = 255  # what floodFill writes into the mask for filled pixels


@dataclasses.dataclass(frozen=True)
class Params:
    """Settings for floor_mask. Defaults were chosen on NYUv2 train only.

    Attributes:
        color_tolerance: Largest LAB difference allowed between a pixel and
            its filled neighbour, per channel.
        canny_low: Canny's lower hysteresis threshold.
        canny_high: Canny's upper hysteresis threshold.
        seed_rows_up: The seed sits this many rows above the bottom edge.
        horizon_row: If set, rows 0 to horizon_row are never floor. Known
            in sim from the camera pitch; unknown for NYUv2.
        blur_kernel: Gaussian blur size, odd.
        close_kernel: Size of the square used to close holes.
    """

    # Chosen by a grid search on NYUv2 train (795 images, Eigen crop):
    # tolerance 1 to 4, Canny (30, 90) to (150, 400). Micro IoU levels off at
    # 0.278 from (80, 200) on; tolerance 1 stops too early, 3+ leaks.
    color_tolerance: int = 2
    canny_low: int = 80
    canny_high: int = 200
    seed_rows_up: int = 5
    horizon_row: int | None = None
    blur_kernel: int = 5
    close_kernel: int = 7


def seed_pixel(height: int, width: int, params: Params) -> tuple[int, int]:
    """Returns the (row, col) the fill starts from: bottom centre."""
    return height - 1 - params.seed_rows_up, width // 2


def floor_mask(rgb: np.ndarray, params: Params | None = None) -> np.ndarray:
    """Estimates which pixels are floor.

    Args:
        rgb: (H, W, 3) uint8 image in RGB order.
        params: Settings; Params() if None.

    Returns:
        (H, W) bool mask. Either empty (the seed sits on an edge or above the
        horizon, so there is no trustworthy start) or it contains the seed.

    Raises:
        ValueError: If the image isn't (H, W, 3) uint8 or is too small to
            hold the seed.
    """
    p = params or Params()
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(
            f"expected (H, W, 3) uint8, got {rgb.shape} {rgb.dtype}"
        )
    h, w = rgb.shape[:2]
    if h <= p.seed_rows_up:
        raise ValueError(f"image height {h} too small for the seed")
    row, col = seed_pixel(h, w, p)

    blurred = cv2.GaussianBlur(rgb, (p.blur_kernel, p.blur_kernel), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, p.canny_low, p.canny_high)
    # Thicken edges so the fill can't slip through one-pixel gaps.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))

    # floodFill's mask is 2 pixels larger than the image (a 1-pixel frame),
    # so image pixel (r, c) is mask pixel (r + 1, c + 1). Nonzero = barrier.
    barrier = np.zeros((h + 2, w + 2), np.uint8)
    barrier[1:-1, 1:-1] = edges > 0
    if p.horizon_row is not None:
        barrier[: p.horizon_row + 2, :] = 1  # image rows 0..horizon_row
    if barrier[row + 1, col]:
        return np.zeros((h, w), dtype=bool)

    tol = (p.color_tolerance,) * 3
    # 4-connected fill, write FILL_VALUE into the mask, leave the image alone.
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (FILL_VALUE << 8)
    cv2.floodFill(lab, barrier, (col, row), (0, 0, 0), tol, tol, flags)
    filled = (barrier[1:-1, 1:-1] == FILL_VALUE).astype(np.uint8)

    # Closing (dilate, then erode) fills small holes and only ever adds
    # pixels, so the seed stays in the mask.
    kernel = np.ones((p.close_kernel, p.close_kernel), np.uint8)
    mask = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel).astype(bool)
    if p.horizon_row is not None:
        mask[: p.horizon_row + 1] = False  # closing may creep up a few rows
    return mask
