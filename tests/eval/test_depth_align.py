"""Tests for minifsd.eval.depth_align: relative disparity to metric depth.

Depth Anything outputs disparity (inverse depth) up to an unknown scale and
shift: d = (1 / depth - t) / s for some s > 0 and t. Fitting s and t by least
squares against ground truth turns it back into metres.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.eval import depth_align, metrics

RNG = np.random.default_rng(0)
GT = RNG.uniform(0.5, 9.5, size=(24, 32))  # metres, NYUv2-like range


def _relative(gt: np.ndarray, scale: float, shift: float) -> np.ndarray:
    """What a relative model would output if it were perfect."""
    return (1.0 / gt - shift) / scale


@given(st.floats(0.05, 20.0), st.floats(-0.08, 0.08))
def test_recovers_the_scale_and_shift(scale: float, shift: float) -> None:
    fit = depth_align.fit(_relative(GT, scale, shift), GT)
    assert fit.scale == pytest.approx(scale, rel=1e-6)
    assert fit.shift == pytest.approx(shift, abs=1e-6)


def test_aligned_depth_matches_ground_truth() -> None:
    disparity = _relative(GT, 3.0, 0.02)
    depth = depth_align.fit(disparity, GT).to_depth(disparity)
    assert depth.dtype == np.float32
    assert metrics.abs_rel(depth, GT) == pytest.approx(0.0, abs=1e-5)


def test_ignores_pixels_without_ground_truth() -> None:
    gt = GT.copy()
    disparity = _relative(gt, 2.0, 0.0)
    gt[:5] = 0.0  # no reading
    disparity[:5] = 1e6  # garbage the fit must not see
    fit = depth_align.fit(disparity, gt)
    assert fit.scale == pytest.approx(2.0)


def test_pooled_fit_over_several_images() -> None:
    pairs = [(_relative(GT * k, 4.0, 0.01), GT * k) for k in (0.5, 0.8, 1.0)]
    fit = depth_align.fit_many(pairs)
    assert fit.scale == pytest.approx(4.0)
    assert fit.shift == pytest.approx(0.01)


def test_non_positive_disparity_becomes_the_far_limit() -> None:
    # A fitted s * d + t at or below zero means "infinitely far"; clamp it
    # to the far limit instead of returning inf or a negative depth.
    fit = depth_align.Fit(scale=1.0, shift=-1.0)
    depth = fit.to_depth(np.array([[0.5, 2.0]]))
    assert depth[0, 0] == pytest.approx(depth_align.MAX_DEPTH)
    assert depth[0, 1] == pytest.approx(1.0)


def test_rejects_too_few_valid_pixels() -> None:
    with pytest.raises(ValueError):
        depth_align.fit(np.ones((2, 2)), np.zeros((2, 2)))


def test_rejects_flat_disparity() -> None:
    # A constant prediction leaves scale and shift undetermined.
    with pytest.raises(ValueError):
        depth_align.fit(np.full(GT.shape, 0.3), GT)
