"""Tests for minifsd.geometry.camera: the pinhole model."""

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
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


# Calibration (V4). A 9 x 6 board of inner corners, 25 mm squares.
PATTERN = (9, 6)
SQUARE_M = 0.025


def _board_poses(
    rng: np.random.Generator, k: frames.Intrinsics, n: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """n (rvec, tvec) poses that put the whole board in view, tilted."""
    board_w = (PATTERN[0] - 1) * SQUARE_M
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    while len(poses) < n:
        # Far enough that the board spans about half the image's width.
        z = k.fx * board_w / (0.5 * k.width) * rng.uniform(0.9, 1.4)
        rvec = rng.uniform(-0.5, 0.5, 3)  # up to about 30 degrees each way
        tvec = np.array(
            [
                rng.uniform(-0.2, 0.05) * z * k.width / k.fx,
                rng.uniform(-0.2, 0.05) * z * k.height / k.fy,
                z,
            ]
        )
        uv = _project_board(rvec, tvec, k)
        inside = (uv >= 0).all() and (uv[:, 0] < k.width - 1).all()
        if inside and (uv[:, 1] < k.height - 1).all():
            poses.append((rvec, tvec))
    return poses


def _project_board(
    rvec: np.ndarray, tvec: np.ndarray, k: frames.Intrinsics
) -> np.ndarray:
    cols, rows = PATTERN
    grid = np.zeros((cols * rows, 3))
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_M
    uv, _ = cv2.projectPoints(grid, rvec, tvec, k.matrix, np.zeros(5))
    return uv.reshape(-1, 2)


@settings(deadline=None, max_examples=15)
@given(
    st.floats(400.0, 1500.0),
    st.floats(0.98, 1.02),
    st.floats(-20.0, 20.0),
    st.floats(-20.0, 20.0),
    st.integers(0, 2**32 - 1),
)
def test_calibration_gives_back_the_camera_that_made_the_corners(
    fx: float, aspect: float, dcx: float, dcy: float, seed: int
) -> None:
    k = frames.Intrinsics(640, 480, fx, fx * aspect, 319.5 + dcx, 239.5 + dcy)
    rng = np.random.default_rng(seed)
    corners = [
        _project_board(rvec, tvec, k) for rvec, tvec in _board_poses(rng, k, 15)
    ]
    found = camera.calibrate(corners, (640, 480), PATTERN, SQUARE_M)
    assert found.rms_px < 0.1  # noiseless corners fit almost exactly
    for name in ("fx", "fy", "cx", "cy"):
        assert getattr(found.k, name) == pytest.approx(
            getattr(k, name), rel=0.01
        ), name
    assert (found.k.width, found.k.height) == (640, 480)


def _board_image() -> tuple[np.ndarray, np.ndarray]:
    """A warped checkerboard and where its inner corners truly are."""
    side, margin = 40, 60
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1  # squares, not corners
    flat = np.full((rows * side + 2 * margin, cols * side + 2 * margin), 255)
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                y, x = margin + j * side, margin + i * side
                flat[y : y + side, x : x + side] = 0
    # Square edges fall between pixels, so a corner is half a pixel before
    # its square's first pixel (pixel centres at whole numbers).
    i, j = np.mgrid[1 : PATTERN[0] + 1, 1 : PATTERN[1] + 1]
    flat_corners = np.stack(
        [margin + i.ravel() * side - 0.5, margin + j.ravel() * side - 0.5], 1
    )
    h = np.array([[0.9, 0.08, 30.0], [-0.05, 0.85, 40.0], [1e-4, 2e-4, 1.0]])
    image = cv2.warpPerspective(
        flat.astype(np.uint8), h, (640, 480), borderValue=255
    )
    true = cv2.perspectiveTransform(flat_corners[None], h)[0]
    return image, true


def test_finds_every_corner_of_a_warped_board_within_half_a_pixel() -> None:
    image, true = _board_image()
    found = camera.find_corners(image, PATTERN)
    assert found is not None
    assert found.shape == (PATTERN[0] * PATTERN[1], 2)
    # OpenCV may number the corners from either end, so match each true
    # corner to its nearest found one.
    gaps = np.linalg.norm(true[:, None] - found[None], axis=2).min(axis=1)
    assert gaps.max() < 0.5


def test_a_blank_image_has_no_corners() -> None:
    blank = np.full((480, 640), 128, dtype=np.uint8)
    assert camera.find_corners(blank, PATTERN) is None
