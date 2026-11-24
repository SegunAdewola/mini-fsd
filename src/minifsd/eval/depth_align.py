"""Fits relative depth to metres: a scale and shift on disparity.

Relative depth models such as Depth Anything V2 output disparity (inverse
depth) that is right only up to an unknown scale s and shift t:
1 / depth = s * d + t. Given ground truth, least squares on the valid pixels
finds s and t. This is the standard zero-shot protocol (MiDaS, Depth
Anything): fitting in disparity space matches what the model predicts.

Two ways to use it, which V5 reports side by side:
- Per image, on that image's own ground truth (fit). This is an oracle: no
  robot has ground truth at run time, so it is an upper bound.
- Once, pooled over training images (fit_many), then applied unchanged to
  every test image. This is what a deployed system could do.
"""

import dataclasses
from collections.abc import Iterable

import numpy as np

# NYUv2's sensor range. Anything the fit puts further away is clamped here.
MAX_DEPTH = 10.0


@dataclasses.dataclass(frozen=True)
class Fit:
    """1 / depth = scale * disparity + shift."""

    scale: float
    shift: float

    def to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """Metric depth, float32, clamped to at most MAX_DEPTH.

        Args:
            disparity: The model's output, any shape.

        Returns:
            Depth in metres, same shape. Where s * d + t is at or below
            1 / MAX_DEPTH (including negative), the depth is MAX_DEPTH.
        """
        inverse = self.scale * np.asarray(disparity, np.float64) + self.shift
        inverse = np.maximum(inverse, 1.0 / MAX_DEPTH)
        return (1.0 / inverse).astype(np.float32)


def _valid_pairs(
    disparity: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if disparity.shape != gt.shape:
        raise ValueError(f"shape mismatch: {disparity.shape} vs {gt.shape}")
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(disparity)
    return (
        np.asarray(disparity[valid], np.float64),
        1.0 / np.asarray(gt[valid], np.float64),
    )


def _solve(d: np.ndarray, target: np.ndarray) -> Fit:
    if d.size < 2:
        raise ValueError("need at least 2 pixels with ground truth")
    if np.ptp(d) == 0:
        raise ValueError("disparity is constant; scale and shift are undefined")
    a = np.stack([d, np.ones_like(d)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(a, target, rcond=None)
    return Fit(float(scale), float(shift))


def fit(disparity: np.ndarray, gt: np.ndarray) -> Fit:
    """Fits scale and shift for one image.

    Args:
        disparity: The model's relative disparity, (H, W).
        gt: Ground-truth depth in metres, (H, W); pixels <= 0 or not finite
            are ignored.

    Returns:
        The least-squares fit in disparity space.

    Raises:
        ValueError: On a shape mismatch, fewer than 2 valid pixels, or a
            constant disparity.
    """
    return _solve(*_valid_pairs(disparity, gt))


def fit_many(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> Fit:
    """Fits one scale and shift over the valid pixels of many images.

    Args:
        pairs: (disparity, gt) per image, as for fit().

    Returns:
        The pooled least-squares fit.
    """
    ds, targets = zip(*(_valid_pairs(d, g) for d, g in pairs), strict=True)
    return _solve(np.concatenate(ds), np.concatenate(targets))
