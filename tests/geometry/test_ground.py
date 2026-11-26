"""Tests for minifsd.geometry.ground: where a pixel's ray meets the floor."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.geometry import camera, frames, ground

K = frames.intrinsics_from_fovy(320, 240, 50.0)


def _pose(height: float, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """The camera's pose in base_link: up `height`, tilted down `pitch` rad.

    base_link's z = 0 is the floor. A rotation about base y by +pitch turns
    the camera's forward axis (base x) down towards the floor.
    """
    c, s = math.cos(pitch), math.sin(pitch)
    tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return tilt @ frames.OPTICAL_TO_BASE, np.array([0.0, 0.0, height])


def test_a_level_camera_sees_a_floor_row_at_h_fy_over_v_minus_cy() -> None:
    rotation, translation = _pose(0.5, 0.0)
    v = np.array([130.0, 160.0, 239.0])
    pixels = np.stack([np.full_like(v, 100.0), v], axis=1)
    depth = ground.floor_depth(pixels, K, rotation, translation)
    np.testing.assert_allclose(depth, 0.5 * K.fy / (v - K.cy), rtol=1e-9)


@given(
    st.floats(0.0, 319.0),
    st.floats(0.0, 239.0),
    st.floats(0.2, 1.6),
    st.floats(0.0, 0.5),  # level to about 30 degrees down
)
def test_a_floor_point_projects_back_to_its_pixel(
    u: float, v: float, height: float, pitch: float
) -> None:
    rotation, translation = _pose(height, pitch)
    pixel = np.array([[u, v]])
    point = ground.floor_points(pixel, K, rotation, translation)[0]
    if np.isnan(point).any():
        return  # at or above the horizon: covered below
    assert point[2] == pytest.approx(0.0, abs=1e-9)  # on the floor
    assert point[0] > 0.0  # ahead of the robot
    optical = rotation.T @ (point - translation)
    np.testing.assert_allclose(
        camera.project(optical[None], K), pixel, atol=1e-6
    )


def test_a_pixel_at_or_above_the_horizon_meets_no_floor() -> None:
    rotation, translation = _pose(0.5, 0.0)
    pixels = np.array([[100.0, 0.0], [100.0, K.cy], [100.0, K.cy + 1.0]])
    depth = ground.floor_depth(pixels, K, rotation, translation)
    points = ground.floor_points(pixels, K, rotation, translation)
    assert np.isnan(depth[:2]).all() and np.isnan(points[:2]).all()
    assert np.isfinite(depth[2]) and np.isfinite(points[2]).all()
