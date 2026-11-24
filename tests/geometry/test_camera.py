"""Tests for minifsd.geometry.camera: the pinhole model."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.geometry import camera, frames

K = frames.intrinsics_from_fovy(320, 240, 50.0)


def test_centre_pixel_is_on_the_optical_axis() -> None:
    depth = np.full((240, 320), 2.0, dtype=np.float32)
    points, pixels = camera.back_project(depth, K)
    # The principal point (159.5, 119.5) sits between four pixels; the
    # pixel at (160, 120) is half a pixel off the axis.
    i = np.flatnonzero((pixels[:, 0] == 160) & (pixels[:, 1] == 120))[0]
    x, y, z = points[i]
    assert z == pytest.approx(2.0)
    assert x == pytest.approx(0.5 * 2.0 / K.fx)
    assert y == pytest.approx(0.5 * 2.0 / K.fy)


@given(
    st.integers(0, 319),
    st.integers(0, 239),
    st.floats(0.1, 20.0),
)
def test_a_pixels_point_projects_back_to_that_pixel(
    u: int, v: int, d: float
) -> None:
    depth = np.zeros((240, 320), dtype=np.float32)
    depth[v, u] = d
    points, pixels = camera.back_project(depth, K)
    assert pixels.tolist() == [[u, v]]
    uv = camera.project(points, K)
    np.testing.assert_allclose(uv, [[u, v]], atol=1e-3)


def test_stride_keeps_every_nth_pixel() -> None:
    depth = np.ones((240, 320), dtype=np.float32)
    points, pixels = camera.back_project(depth, K, stride=4)
    assert len(points) == (240 // 4) * (320 // 4)
    assert set(np.unique(pixels[:, 0] % 4)) == {0}


def test_drops_pixels_without_depth() -> None:
    depth = np.ones((240, 320), dtype=np.float32)
    depth[0, 0], depth[0, 1], depth[0, 2] = 0.0, np.nan, np.inf
    points, _ = camera.back_project(depth, K)
    assert len(points) == 240 * 320 - 3
    assert np.isfinite(points).all()


def test_rejects_a_depth_image_of_the_wrong_size() -> None:
    with pytest.raises(ValueError):
        camera.back_project(np.ones((10, 10), dtype=np.float32), K)
