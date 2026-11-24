"""The pinhole camera model: pixels and depth to 3D points and back.

Points are in the camera's optical frame (x right, y down, z ahead; R4), and
pixel centres sit at whole numbers (OpenCV and ROS). Depth is distance along
the viewing axis (z), as MuJoCo, NYUv2 and the depth models give it.

This is the projection half of V4. Calibration (finding the intrinsics of a
real camera from checkerboard photos) comes with V4.
"""

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
