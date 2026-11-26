"""Tests for minifsd.perception.metric_depth: scale from the floor (M4.2)."""

import math

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from minifsd.eval import depth_align
from minifsd.geometry import frames, ground
from minifsd.perception import metric_depth

K = frames.intrinsics_from_fovy(320, 240, 50.0)
FALLBACK = depth_align.Fit(scale=0.5, shift=0.1)


def _pose(height: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera up `height` in base_link, tilted down `pitch` rad."""
    c, s = math.cos(pitch), math.sin(pitch)
    tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return tilt @ frames.OPTICAL_TO_BASE, np.array([0.0, 0.0, height])


def _room(
    height: float, pitch: float, wall: float
) -> tuple[np.ndarray, np.ndarray]:
    """True depth and floor mask: a floor, and a wall `wall` metres ahead."""
    rotation, translation = _pose(height, pitch)
    v, u = np.mgrid[0 : K.height, 0 : K.width]
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64)
    floor = ground.floor_depth(pixels, K, rotation, translation)
    # The wall is the plane x = wall in base_link; t is depth along the axis.
    rays = np.stack(
        [
            (pixels[:, 0] - K.cx) / K.fx,
            (pixels[:, 1] - K.cy) / K.fy,
            np.ones(len(pixels)),
        ],
        axis=1,
    )
    dx = (rays @ rotation.T)[:, 0]
    to_wall = np.where(dx > 0, (wall - translation[0]) / dx, np.inf)
    sees_floor = np.isfinite(floor) & (floor < to_wall)
    depth = np.where(sees_floor, floor, to_wall)
    shape = (K.height, K.width)
    return depth.reshape(shape), sees_floor.reshape(shape)


@settings(deadline=None, max_examples=25)
@given(
    st.floats(0.3, 1.0),  # camera height, m
    st.floats(0.0, 0.3),  # pitch down, rad
    st.floats(1.5, 6.0),  # wall distance, m
    st.floats(0.1, 10.0),  # the model's unknown scale (positive)
    st.floats(-1.0, 1.0),  # and shift
)
def test_any_positive_scale_and_shift_comes_back_as_metres(
    height: float, pitch: float, wall: float, a: float, b: float
) -> None:
    depth, floor = _room(height, pitch, wall)
    # A high, level camera close to a wall may see no floor at all; that is
    # the fallback's case, tested below.
    assume(floor.sum() >= metric_depth.MIN_FLOOR_PIXELS)
    disparity = a / depth + b  # what a relative model might output
    rotation, translation = _pose(height, pitch)
    out = metric_depth.metric_depth(
        disparity, floor, K, rotation, translation, FALLBACK
    )
    assert out.from_floor
    near = depth < depth_align.MAX_DEPTH  # beyond it to_depth clamps
    np.testing.assert_allclose(out.depth[near], depth[near], rtol=0.01)


def test_too_little_floor_in_view_uses_the_fallback() -> None:
    depth, _ = _room(0.5, 0.1, 3.0)
    disparity = 2.0 / depth
    rotation, translation = _pose(0.5, 0.1)
    no_floor = np.zeros_like(disparity, dtype=bool)
    out = metric_depth.metric_depth(
        disparity, no_floor, K, rotation, translation, FALLBACK
    )
    assert not out.from_floor
    assert out.fit == FALLBACK
    np.testing.assert_allclose(out.depth, FALLBACK.to_depth(disparity))


def test_an_inverted_fit_on_the_floor_is_not_trusted() -> None:
    # Disparity that rises with distance gives a negative scale; a wrong
    # floor mask can do the same. Metres from it would be nonsense.
    depth, floor = _room(0.5, 0.1, 3.0)
    disparity = depth.copy()
    rotation, translation = _pose(0.5, 0.1)
    out = metric_depth.metric_depth(
        disparity, floor, K, rotation, translation, FALLBACK
    )
    assert not out.from_floor
    assert out.fit == FALLBACK
