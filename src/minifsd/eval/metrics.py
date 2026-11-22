"""Metrics for floor segmentation and monocular depth.

Every experiment table (E1 to E6) rests on these functions, so they are
tested with both hand-computed cases and Hypothesis properties.

Floor segmentation uses intersection over union (IoU). Depth uses the two
metrics from Eigen et al. (2014) that every monocular depth paper reports:
AbsRel (mean relative error) and delta1 (share of pixels within 25%).
"""

import numpy as np

DELTA1_THRESHOLD = 1.25


def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


def _valid_depth(gt: np.ndarray) -> np.ndarray:
    """Returns a mask of pixels with usable ground-truth depth."""
    return np.isfinite(gt) & (gt > 0)


def intersection_union(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None
) -> tuple[int, int]:
    """Counts intersection and union pixels of two boolean masks.

    Returning counts (not a ratio) lets a dataset's IoU be computed as
    total intersection over total union across all images.

    Args:
        pred: Predicted floor mask, boolean, shape (H, W).
        gt: Ground-truth floor mask, boolean, same shape.
        valid: Optional boolean mask; only True pixels are counted.

    Returns:
        (intersection, union) as pixel counts.

    Raises:
        ValueError: If the shapes differ.
    """
    _check_same_shape(pred, gt)
    if valid is not None:
        _check_same_shape(pred, valid)
    if valid is not None:
        # Unlabeled pixels count as neither floor nor not-floor.
        pred = pred & valid
        gt = gt & valid
    inter = int(np.count_nonzero(pred & gt))
    union = int(np.count_nonzero(pred | gt))
    return inter, union


def floor_iou(
    pred: np.ndarray, gt: np.ndarray, valid: np.ndarray | None = None
) -> float:
    """Intersection over union of two floor masks for one image.

    Args:
        pred: Predicted floor mask, boolean, shape (H, W).
        gt: Ground-truth floor mask, boolean, same shape.
        valid: Optional boolean mask; only True pixels are counted.

    Returns:
        IoU in [0, 1]. Defined as 1.0 when both masks are empty: there was
        no floor to find and none was predicted.

    Raises:
        ValueError: If the shapes differ.
    """
    inter, union = intersection_union(pred, gt, valid)
    if union == 0:
        return 1.0
    return inter / union


def abs_rel(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute relative depth error, mean(|pred - gt| / gt).

    Only pixels with valid ground truth (finite and > 0) are scored.

    Args:
        pred: Predicted depth in metres, shape (H, W).
        gt: Ground-truth depth in metres, same shape.

    Returns:
        AbsRel, 0.0 for a perfect prediction. Lower is better.

    Raises:
        ValueError: If the shapes differ or no pixel has valid ground truth.
    """
    _check_same_shape(pred, gt)
    valid = _valid_depth(gt)
    if not valid.any():
        raise ValueError("no pixels with valid ground-truth depth")
    # Divide by the true depth, so the error is relative to reality. Dividing
    # by pred would reward a model for predicting everything too far away.
    p, g = pred[valid], gt[valid]
    return float(np.mean(np.abs(p - g) / g))


def delta1(
    pred: np.ndarray, gt: np.ndarray, threshold: float = DELTA1_THRESHOLD
) -> float:
    """Share of pixels where max(pred / gt, gt / pred) < threshold.

    Only pixels with valid ground truth are scored. A non-positive or
    non-finite prediction on a valid pixel counts as wrong.

    Args:
        pred: Predicted depth in metres, shape (H, W).
        gt: Ground-truth depth in metres, same shape.
        threshold: Ratio limit; 1.25 is the standard delta1.

    Returns:
        A fraction in [0, 1]. Higher is better.

    Raises:
        ValueError: If the shapes differ or no pixel has valid ground truth.
    """
    _check_same_shape(pred, gt)
    valid = _valid_depth(gt)
    if not valid.any():
        raise ValueError("no pixels with valid ground-truth depth")
    p, g = pred[valid], gt[valid]
    usable = np.isfinite(p) & (p > 0)
    # Only usable predictions can pass; the rest stay False (wrong).
    within = np.zeros(p.shape, dtype=bool)
    ratio = np.maximum(p[usable] / g[usable], g[usable] / p[usable])
    within[usable] = ratio < threshold
    return float(np.mean(within))
