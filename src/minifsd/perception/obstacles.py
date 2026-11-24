"""Depth image to obstacle points in the robot frame (R10).

The perception stub: the camera's depth becomes a point cloud that Nav2's
local costmap marks as obstacles. Points near the floor are dropped (the
floor is not an obstacle), and so are points above the robot's reach and
beyond a range where depth stops being trustworthy. In M4 the input becomes
the model's depth instead of the sim's ground truth; this code stays.
"""

import numpy as np

from minifsd.geometry import camera, frames

MIN_HEIGHT = 0.05  # metres above the floor; below this is floor
MAX_HEIGHT = 1.0  # above the robot (0.34 m tall) with room for overhangs
MAX_RANGE = 5.0  # horizontal distance from the robot, metres
STRIDE = 4  # every 4th pixel: 4,800 points from 320 x 240


def obstacle_points(
    depth: np.ndarray,
    k: frames.Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
    stride: int = STRIDE,
) -> np.ndarray:
    """The points of a depth image that are obstacles, in base_link.

    Args:
        depth: (H, W) metres along the viewing axis.
        k: The camera's intrinsics.
        rotation: (3, 3) optical frame to base_link (from TF).
        translation: (3,) the camera's position in base_link (from TF).
        stride: Pixel step, as for camera.back_project.

    Returns:
        (N, 3) float32 points in base_link (x ahead, y left, z up).
    """
    points, _ = camera.back_project(depth, k, stride)
    base = points @ np.asarray(rotation).T + np.asarray(translation)
    height = base[:, 2]
    reach = np.hypot(base[:, 0], base[:, 1])
    keep = (
        (height >= MIN_HEIGHT) & (height <= MAX_HEIGHT) & (reach <= MAX_RANGE)
    )
    return np.asarray(base[keep], dtype=np.float32)
