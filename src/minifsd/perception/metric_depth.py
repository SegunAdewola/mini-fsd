"""Metric depth from a relative model, scaled on the floor each frame (M4.2).

The depth student outputs disparity right only up to a scale and shift,
1 / depth = s * d + t (minifsd.eval.depth_align). E3 fitted s and t once
over many images, and that global fit left AbsRel near 0.24 because the
scale drifts from frame to frame; it also comes from NYUv2, not the sim.

The floor gives every frame its own fit. The camera's height and pitch fix
the depth of each floor pixel by geometry alone (minifsd.geometry.ground),
and the floor model says which pixels are floor. Least squares on those
pixels finds this frame's s and t, which then apply to the whole image.

Two cases fall back to a fixed fit: too little floor in view (facing a wall
up close), and a fit whose scale isn't positive. A wrong floor mask, say a
counter marked as floor, can pull the fit that way, and a negative scale
would put near things far away (the E3b sign lesson).
"""

import dataclasses

import numpy as np

from minifsd.eval import depth_align
from minifsd.geometry import frames, ground

# About 0.7% of a 320 x 240 image: a strip of floor a few rows tall. Fewer
# pixels than this span too little depth for the fit to be steady.
MIN_FLOOR_PIXELS = 500


@dataclasses.dataclass(frozen=True)
class Metric:
    """One frame's metric depth and where its scale came from.

    Attributes:
        depth: (H, W) float32 metres, clamped as by Fit.to_depth.
        fit: The scale and shift used.
        from_floor: True if fitted on this frame's floor, False if the
            fallback was used.
    """

    depth: np.ndarray
    fit: depth_align.Fit
    from_floor: bool


def metric_depth(
    disparity: np.ndarray,
    floor: np.ndarray,
    k: frames.Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
    fallback: depth_align.Fit,
    min_floor: int = MIN_FLOOR_PIXELS,
) -> Metric:
    """Scales a relative disparity map to metres using the visible floor.

    Args:
        disparity: (H, W) the model's relative disparity (larger is closer).
        floor: (H, W) bool, the floor model's mask.
        k: The camera's intrinsics, for the same (H, W).
        rotation: (3, 3) optical frame to base_link (from TF).
        translation: (3,) the camera's position in base_link (from TF).
        fallback: The fit to use when the floor can't give one.
        min_floor: Fewest usable floor pixels for a per-frame fit.

    Returns:
        The metric depth, the fit and whether it came from the floor.

    Raises:
        ValueError: If disparity, floor and intrinsics disagree on size.
    """
    shape = (k.height, k.width)
    if disparity.shape != shape or floor.shape != shape:
        raise ValueError(
            f"disparity {disparity.shape} and floor {floor.shape} must both "
            f"be {shape}"
        )
    v, u = np.mgrid[0 : k.height, 0 : k.width]
    pixels = np.stack([u.ravel(), v.ravel()], axis=1)
    known = ground.floor_depth(pixels, k, rotation, translation).reshape(shape)
    use = floor & np.isfinite(known) & np.isfinite(disparity)
    if int(use.sum()) >= min_floor and np.ptp(disparity[use]) > 0:
        fit = depth_align.fit(disparity, np.where(use, known, 0.0))
        if fit.scale > 0:
            return Metric(fit.to_depth(disparity), fit, from_floor=True)
    return Metric(fallback.to_depth(disparity), fallback, from_floor=False)
