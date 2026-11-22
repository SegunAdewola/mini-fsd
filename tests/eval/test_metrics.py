"""Tests for minifsd.eval.metrics.

Hand-computed cases pin exact values. Hypothesis properties pin the rules
that must hold for any input. See study guide 3.5 for the definitions.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from minifsd.eval import metrics

# --- Strategies -------------------------------------------------------------

SHAPES = hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=8)


@st.composite
def mask_pairs(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    """Two boolean masks with the same random shape."""
    shape = draw(SHAPES)
    a = draw(hnp.arrays(np.bool_, shape))
    b = draw(hnp.arrays(np.bool_, shape))
    return a, b


@st.composite
def depth_maps(draw: st.DrawFn) -> np.ndarray:
    """A depth map in metres, all values valid (0.1 m to 10 m)."""
    shape = draw(SHAPES)
    return draw(hnp.arrays(np.float64, shape, elements=st.floats(0.1, 10.0)))


def _rows_mask(rows: int, shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[:rows] = True
    return m


# --- IoU: hand-computed ---------------------------------------------------


def test_iou_partial_overlap() -> None:
    gt = _rows_mask(5)  # 50 floor pixels
    pred = _rows_mask(4)  # 40 predicted, all inside gt
    assert metrics.floor_iou(pred, gt) == pytest.approx(40 / 50)


def test_intersection_union_counts() -> None:
    gt = _rows_mask(5)
    pred = _rows_mask(4)
    assert metrics.intersection_union(pred, gt) == (40, 50)


def test_iou_ignores_pixels_outside_valid() -> None:
    gt = _rows_mask(5)
    pred = _rows_mask(4)
    valid = np.ones((10, 10), dtype=bool)
    valid[4] = False  # row 4 is unlabeled; it is the only disagreement
    assert metrics.floor_iou(pred, gt, valid) == pytest.approx(1.0)


def test_iou_both_empty_is_perfect() -> None:
    empty = np.zeros((4, 4), dtype=bool)
    assert metrics.floor_iou(empty, empty) == 1.0


def test_iou_empty_prediction_scores_zero() -> None:
    assert metrics.floor_iou(np.zeros((4, 4), bool), _rows_mask(2, (4, 4))) == 0


def test_iou_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.floor_iou(np.zeros((4, 4), bool), np.zeros((4, 5), bool))


# --- IoU: properties ------------------------------------------------------


@given(mask_pairs())
def test_iou_is_between_zero_and_one(
    masks: tuple[np.ndarray, np.ndarray],
) -> None:
    assert 0.0 <= metrics.floor_iou(*masks) <= 1.0


@given(mask_pairs())
def test_iou_is_symmetric(masks: tuple[np.ndarray, np.ndarray]) -> None:
    a, b = masks
    assert metrics.floor_iou(a, b) == metrics.floor_iou(b, a)


@given(mask_pairs())
def test_iou_of_mask_with_itself_is_one(
    masks: tuple[np.ndarray, np.ndarray],
) -> None:
    a, _ = masks
    assert metrics.floor_iou(a, a) == 1.0


@given(mask_pairs())
def test_intersection_never_exceeds_union(
    masks: tuple[np.ndarray, np.ndarray],
) -> None:
    inter, union = metrics.intersection_union(*masks)
    assert 0 <= inter <= union


# --- Depth: hand-computed -------------------------------------------------


def test_abs_rel_ten_percent_too_far() -> None:
    gt = np.full((3, 3), 2.0)
    assert metrics.abs_rel(gt * 1.1, gt) == pytest.approx(0.1)


def test_abs_rel_skips_invalid_ground_truth() -> None:
    gt = np.array([[2.0, 0.0, np.nan, np.inf]])
    pred = np.array([[2.2, 5.0, 5.0, 5.0]])  # only the first pixel counts
    assert metrics.abs_rel(pred, gt) == pytest.approx(0.1)


def test_abs_rel_rejects_no_valid_pixels() -> None:
    with pytest.raises(ValueError):
        metrics.abs_rel(np.ones((2, 2)), np.zeros((2, 2)))


def test_delta1_counts_ratios_below_threshold() -> None:
    gt = np.full((1, 4), 2.0)
    pred = gt * np.array([1.2, 1 / 1.2, 1.3, 1 / 1.3])  # 2 inside, 2 outside
    assert metrics.delta1(pred, gt) == pytest.approx(0.5)


def test_delta1_treats_non_positive_prediction_as_wrong() -> None:
    gt = np.full((1, 2), 2.0)
    pred = np.array([[2.0, 0.0]])
    assert metrics.delta1(pred, gt) == pytest.approx(0.5)


def test_depth_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        metrics.abs_rel(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError):
        metrics.delta1(np.ones((2, 2)), np.ones((2, 3)))


# --- Depth: properties ----------------------------------------------------


@given(depth_maps())
def test_perfect_depth_scores_perfectly(gt: np.ndarray) -> None:
    assert metrics.abs_rel(gt, gt) == pytest.approx(0.0)
    assert metrics.delta1(gt, gt) == 1.0


@given(depth_maps(), depth_maps())
def test_delta1_is_symmetric_in_pred_and_gt(
    a: np.ndarray, b: np.ndarray
) -> None:
    if a.shape != b.shape:
        b = np.resize(b, a.shape)
    assert metrics.delta1(a, b) == pytest.approx(metrics.delta1(b, a))


@given(depth_maps(), st.floats(1 / 1.24, 1.24))
def test_uniform_scale_inside_threshold_keeps_delta1_perfect(
    gt: np.ndarray, scale: float
) -> None:
    assert metrics.delta1(gt * scale, gt) == 1.0


@given(depth_maps(), st.floats(0.5, 2.0))
def test_abs_rel_of_uniform_scale_is_scale_error(
    gt: np.ndarray, scale: float
) -> None:
    assert metrics.abs_rel(gt * scale, gt) == pytest.approx(abs(scale - 1))
