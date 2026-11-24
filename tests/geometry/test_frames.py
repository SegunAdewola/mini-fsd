"""Tests for minifsd.geometry.frames: quaternions, frames, intrinsics.

ROS conventions (REP 103 and REP 105): quaternions are (x, y, z, w); the base
frame is x forward, y left, z up; a camera's optical frame is x right, y down,
z forward.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.geometry import frames

YAWS = st.floats(-10 * math.pi, 10 * math.pi)


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _same_rotation(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    # q and -q are the same rotation, so compare up to sign.
    return bool(np.allclose(a, b, atol=1e-9) or np.allclose(a, -np.array(b)))


def test_zero_yaw_is_the_identity_quaternion() -> None:
    assert frames.yaw_to_quaternion(0.0) == pytest.approx((0, 0, 0, 1))


def test_quarter_turn() -> None:
    h = math.sqrt(0.5)
    assert frames.yaw_to_quaternion(math.pi / 2) == pytest.approx((0, 0, h, h))


@given(YAWS)
def test_quaternion_is_unit_length(yaw: float) -> None:
    q = frames.yaw_to_quaternion(yaw)
    assert math.hypot(*q) == pytest.approx(1.0)


@given(YAWS)
def test_yaw_round_trips(yaw: float) -> None:
    back = frames.quaternion_to_yaw(frames.yaw_to_quaternion(yaw))
    assert -math.pi <= back <= math.pi
    # Equal as angles: the difference is a whole number of turns.
    diff = math.remainder(back - yaw, 2 * math.pi)
    assert diff == pytest.approx(0.0, abs=1e-9)


@given(YAWS)
def test_quaternion_from_matrix_agrees_with_yaw(yaw: float) -> None:
    q = frames.quaternion_from_matrix(_yaw_matrix(yaw))
    assert _same_rotation(q, frames.yaw_to_quaternion(yaw))


@pytest.mark.parametrize(
    ("optical", "base"),
    [
        ((0, 0, 1), (1, 0, 0)),  # optical z (forward) is base x (forward)
        ((1, 0, 0), (0, -1, 0)),  # optical x (image right) is base -y (right)
        ((0, 1, 0), (0, 0, -1)),  # optical y (image down) is base -z (down)
    ],
)
def test_optical_axes_map_to_base_axes(
    optical: tuple[int, int, int], base: tuple[int, int, int]
) -> None:
    np.testing.assert_allclose(frames.OPTICAL_TO_BASE @ optical, base)


def test_optical_rotation_is_a_proper_rotation() -> None:
    r = frames.OPTICAL_TO_BASE
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_optical_quaternion_is_the_ros_standard() -> None:
    # The usual static TF for a camera: roll -90 deg, then yaw -90 deg,
    # which is the quaternion (-0.5, 0.5, -0.5, 0.5).
    assert _same_rotation(frames.OPTICAL_TO_BASE_QUAT, (-0.5, 0.5, -0.5, 0.5))


def test_fy_for_240_pixels_at_50_degrees() -> None:
    k = frames.intrinsics_from_fovy(320, 240, 50.0)
    expected = 120 / math.tan(math.radians(25))  # about 257.3
    assert k.fy == pytest.approx(expected)
    assert k.fx == pytest.approx(k.fy)  # square pixels


def test_principal_point_is_the_image_centre() -> None:
    # Pixel centres sit at whole numbers (OpenCV and ROS), so the centre of
    # a 320 px wide image is at 159.5, between pixels 159 and 160.
    k = frames.intrinsics_from_fovy(320, 240, 50.0)
    assert (k.cx, k.cy) == (159.5, 119.5)


def test_k_matrix_layout() -> None:
    k = frames.intrinsics_from_fovy(320, 240, 50.0)
    expected = np.array([[k.fx, 0, k.cx], [0, k.fy, k.cy], [0, 0, 1]])
    np.testing.assert_allclose(k.matrix, expected)


@given(
    st.integers(min_value=16, max_value=2000),
    st.integers(min_value=16, max_value=2000),
    st.floats(min_value=10, max_value=150),
)
def test_field_of_view_edges_land_on_the_image_edges(
    width: int, height: int, fovy_deg: float
) -> None:
    # A ray at the top edge of the field of view hits the top edge of the
    # top row of pixels, which is v = -0.5; the bottom edge is height - 0.5.
    k = frames.intrinsics_from_fovy(width, height, fovy_deg)
    t = math.tan(math.radians(fovy_deg) / 2)
    assert k.cy - k.fy * t == pytest.approx(-0.5)
    assert k.cy + k.fy * t == pytest.approx(height - 0.5)


def test_rejects_bad_field_of_view() -> None:
    with pytest.raises(ValueError):
        frames.intrinsics_from_fovy(320, 240, 0.0)
    with pytest.raises(ValueError):
        frames.intrinsics_from_fovy(320, 240, 180.0)


@given(YAWS)
def test_matrix_round_trips_through_a_quaternion(yaw: float) -> None:
    m = frames.quaternion_to_matrix(frames.yaw_to_quaternion(yaw))
    np.testing.assert_allclose(m, _yaw_matrix(yaw), atol=1e-9)


def test_optical_quaternion_gives_the_optical_matrix() -> None:
    np.testing.assert_allclose(
        frames.quaternion_to_matrix(frames.OPTICAL_TO_BASE_QUAT),
        frames.OPTICAL_TO_BASE,
        atol=1e-12,
    )
