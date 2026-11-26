"""The pinhole camera model: pixels and depth to 3D points and back.

Points are in the camera's optical frame (x right, y down, z ahead; R4), and
pixel centres sit at whole numbers (OpenCV and ROS). Depth is distance along
the viewing axis (z), as MuJoCo, NYUv2 and the depth models give it.

Calibration (V4) finds a real camera's intrinsics and lens distortion from
photos of a printed checkerboard, with OpenCV.
"""

import dataclasses
from collections.abc import Sequence

import cv2
import numpy as np

from minifsd.geometry import frames


def back_project(
    depth: np.ndarray, k: frames.Intrinsics, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Lifts pixels with depth into 3D.

    A pixel (u, v) with depth d is the point
    ((u - cx) d / fx, (v - cy) d / fy, d).

    Args:
        depth: (H, W) metres along the viewing axis; 0, NaN and inf mean
            "no reading" and are skipped.
        k: The camera's intrinsics; their size must match the image.
        stride: Use every stride-th pixel in each direction.

    Returns:
        (points, pixels): (N, 3) float64 points in the optical frame, and the
        (N, 2) int (u, v) pixel each came from.

    Raises:
        ValueError: If the image size doesn't match the intrinsics.
    """
    if depth.shape != (k.height, k.width):
        raise ValueError(
            f"depth is {depth.shape}, intrinsics are for {(k.height, k.width)}"
        )
    v, u = np.mgrid[0 : k.height : stride, 0 : k.width : stride]
    d = np.asarray(depth[v, u], dtype=np.float64)
    valid = np.isfinite(d) & (d > 0)
    u, v, d = u[valid], v[valid], d[valid]
    points = np.stack([(u - k.cx) * d / k.fx, (v - k.cy) * d / k.fy, d], axis=1)
    return points, np.stack([u, v], axis=1)


def project(points: np.ndarray, k: frames.Intrinsics) -> np.ndarray:
    """Projects optical-frame points to pixel coordinates.

    Args:
        points: (N, 3), with z > 0 (in front of the camera).
        k: The intrinsics.

    Returns:
        (N, 2) float (u, v).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return np.stack([k.fx * x / z + k.cx, k.fy * y / z + k.cy], axis=1)


# Inner corners (columns, rows) of OpenCV's standard board: 10 x 7 squares.
BOARD = (9, 6)


@dataclasses.dataclass(frozen=True)
class Calibration:
    """A calibrated camera.

    Attributes:
        k: Intrinsics at the calibration photos' size.
        distortion: OpenCV's (k1, k2, p1, p2, k3).
        rms_px: Root-mean-square reprojection error over all corners.
    """

    k: frames.Intrinsics
    distortion: np.ndarray
    rms_px: float


def board_points(pattern: tuple[int, int], square_m: float) -> np.ndarray:
    """The board's inner corners in its own plane (z = 0), in metres.

    In OpenCV's order: along a row first (x), then row by row (y), which is
    the order find_corners returns them in.
    """
    cols, rows = pattern
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return points


def find_corners(
    gray: np.ndarray, pattern: tuple[int, int] = BOARD
) -> np.ndarray | None:
    """Finds a checkerboard's inner corners to a fraction of a pixel.

    Args:
        gray: (H, W) uint8 image.
        pattern: Inner corners (columns, rows).

    Returns:
        (columns * rows, 2) float64 (u, v), or None if the whole board isn't
        found.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not ok:
        return None
    # The detector lands within about a pixel; cornerSubPix then moves each
    # corner to where the image gradients around it meet.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    return np.asarray(refined, np.float64).reshape(-1, 2)


def calibrate(
    corners: Sequence[np.ndarray],
    image_size: tuple[int, int],
    pattern: tuple[int, int] = BOARD,
    square_m: float = 0.025,
) -> Calibration:
    """Intrinsics and distortion from the corners of several board photos.

    Args:
        corners: Per photo, find_corners' (N, 2) output. Use 15 or more
            photos with the board tilted and in every part of the frame.
        image_size: (width, height) of the photos.
        pattern: Inner corners (columns, rows).
        square_m: Side of one square, measured on the print.

    Returns:
        The calibration. A wrong square size scales only the board poses,
        which aren't returned, so the intrinsics don't depend on it.
    """
    board = board_points(pattern, square_m)
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        [board] * len(corners),
        [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in corners],
        image_size,
        None,
        None,
    )
    width, height = image_size
    k = frames.Intrinsics(
        width,
        height,
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )
    return Calibration(k, np.asarray(distortion).ravel(), float(rms))
