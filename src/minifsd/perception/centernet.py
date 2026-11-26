"""CenterNet-style box targets and decoding for HydraNet-lite (H1, E4).

A detector with no anchors and no non-maximum suppression (Zhou et al.,
"Objects as Points", 2019): each object is the peak of a heatmap at its box
centre, one heatmap per class, plus the box's size and the centre's offset
within its grid cell. Decoding takes the local maxima of the heatmaps.

The grid is the input divided by STRIDE: the DINOv2 patch grid (14 px)
upsampled twice, so a 238 x 322 input gives 34 x 46 cells. Sizes are in
cells, offsets in fractions of a cell, boxes in input pixels.

numpy and scipy only, so the targets are built in the data loader and the
decoding runs anywhere, including after ONNX export (E6).
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

STRIDE = 7  # input pixels per grid cell
MIN_OVERLAP = 0.7  # CenterNet's IoU for the Gaussian's radius
TOP_K = 100  # peaks kept per image


@dataclasses.dataclass(frozen=True)
class Box:
    """A box in input pixels; x, y the top-left corner."""

    cls: int
    x: float
    y: float
    w: float
    h: float


@dataclasses.dataclass(frozen=True)
class Found(Box):
    """A decoded box with its heatmap score."""

    score: float = 0.0


@dataclasses.dataclass(frozen=True, eq=False)
class Targets:
    """What the head learns, for one image."""

    heatmap: np.ndarray  # (C, H, W) float32 in [0, 1]
    size: np.ndarray  # (2, H, W) box width and height in cells
    offset: np.ndarray  # (2, H, W) centre within its cell, in [0, 1)
    mask: np.ndarray  # (H, W) bool: cells holding a box centre


def gaussian_radius(h: float, w: float) -> int:
    """CenterNet's radius: how far a corner may move and keep MIN_OVERLAP.

    The smallest root of the three cases in Law and Deng's CornerNet, as
    CenterNet's code computes it; h and w in cells.
    """
    o = MIN_OVERLAP
    b1, c1 = h + w, w * h * (1 - o) / (1 + o)
    r1 = (b1 + math.sqrt(b1**2 - 4 * c1)) / 2
    b2, c2 = 2 * (h + w), (1 - o) * w * h
    r2 = (b2 + math.sqrt(b2**2 - 16 * c2)) / 2
    a3, b3, c3 = 4 * o, -2 * o * (h + w), (o - 1) * w * h
    r3 = (b3 + math.sqrt(b3**2 - 4 * a3 * c3)) / 2
    return max(0, int(min(r1, r2, r3)))


def encode(
    boxes: Sequence[Box], grid: tuple[int, int], n_classes: int
) -> Targets:
    """Heatmaps, sizes and offsets for one image's boxes.

    Args:
        boxes: Boxes in input pixels.
        grid: (rows, cols) of the output grid.
        n_classes: Heatmap channels.
    """
    rows, cols = grid
    heatmap = np.zeros((n_classes, rows, cols), np.float32)
    size = np.zeros((2, rows, cols), np.float32)
    offset = np.zeros((2, rows, cols), np.float32)
    mask = np.zeros((rows, cols), bool)
    ys, xs = np.mgrid[0:rows, 0:cols]
    for box in boxes:
        cx = (box.x + box.w / 2) / STRIDE
        cy = (box.y + box.h / 2) / STRIDE
        col = min(max(int(cx), 0), cols - 1)
        row = min(max(int(cy), 0), rows - 1)
        radius = gaussian_radius(box.h / STRIDE, box.w / STRIDE)
        sigma = (2 * radius + 1) / 6
        blob = np.exp(-((xs - col) ** 2 + (ys - row) ** 2) / (2 * sigma**2))
        blob[np.abs(xs - col) > radius] = 0
        blob[np.abs(ys - row) > radius] = 0
        heatmap[box.cls] = np.maximum(heatmap[box.cls], blob)
        size[:, row, col] = (box.w / STRIDE, box.h / STRIDE)
        offset[:, row, col] = (cx - col, cy - row)
        mask[row, col] = True
    return Targets(heatmap, size, offset, mask)


def decode(
    heatmap: np.ndarray,
    size: np.ndarray,
    offset: np.ndarray,
    min_score: float = 0.001,
    top_k: int = TOP_K,
) -> list[Found]:
    """Boxes at the heatmaps' local maxima, best first.

    Args:
        heatmap: (C, H, W) scores in [0, 1] (after the sigmoid).
        size: (2, H, W) width and height in cells.
        offset: (2, H, W) centre offsets within cells.
        min_score: Peaks below this are dropped.
        top_k: At most this many boxes.
    """
    peaks = heatmap == ndimage.maximum_filter(heatmap, size=(1, 3, 3))
    peaks &= heatmap >= min_score
    peaks &= heatmap > 0
    cls, row, col = np.nonzero(peaks)
    scores = heatmap[cls, row, col]
    order = np.argsort(-scores, kind="stable")[:top_k]
    found = []
    for i in order:
        c, r, q = int(cls[i]), int(row[i]), int(col[i])
        w = float(size[0, r, q]) * STRIDE
        h = float(size[1, r, q]) * STRIDE
        cx = (q + float(offset[0, r, q])) * STRIDE
        cy = (r + float(offset[1, r, q])) * STRIDE
        found.append(Found(c, cx - w / 2, cy - h / 2, w, h, float(scores[i])))
    return found
