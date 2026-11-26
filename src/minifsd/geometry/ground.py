"""Where a pixel's viewing ray meets the floor (M4.1).

base_link's z = 0 is the floor. A pixel (u, v) looks along the optical-frame
ray r = ((u - cx) / fx, (v - cy) / fy, 1); in base_link that is d = R r from
the camera's position c. The ray meets the floor where c_z + t d_z = 0, so
t = -c_z / d_z, and since r's optical z is 1, t is also the pixel's depth
along the viewing axis, the same quantity a depth image holds. A ray that
doesn't point down (d_z >= 0) is at or above the horizon and never meets
the floor.

M4.2 fits each frame's depth scale on floor pixels against this depth, and
M4.5 puts a box's bottom edge on the floor with floor_points.
"""

import numpy as np

from minifsd.geometry import frames


def floor_depth(
    pixels: np.ndarray,
    k: frames.Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Depth along the viewing axis at which each pixel's ray meets the floor.

    Args:
        pixels: (N, 2) float (u, v).
        k: The camera's intrinsics.
        rotation: (3, 3) optical frame to base_link (from TF).
        translation: (3,) the camera's position in base_link (from TF).

    Returns:
        (N,) float64 metres; NaN where the ray never meets the floor.
    """
    d = _rays(pixels, k) @ np.asarray(rotation, np.float64).T
    height = float(np.asarray(translation)[2])
    down = d[:, 2] < 0.0
    t = np.full(len(d), np.nan)
    t[down] = -height / d[down, 2]
    return t


def floor_points(
    pixels: np.ndarray,
    k: frames.Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """The floor point each pixel sees, in base_link.

    Args:
        pixels: (N, 2) float (u, v).
        k: The camera's intrinsics.
        rotation: (3, 3) optical frame to base_link (from TF).
        translation: (3,) the camera's position in base_link (from TF).

    Returns:
        (N, 3) float64 points with z = 0; rows of NaN where the ray never
        meets the floor.
    """
    d = _rays(pixels, k) @ np.asarray(rotation, np.float64).T
    t = floor_depth(pixels, k, rotation, translation)
    points: np.ndarray = np.asarray(translation, np.float64) + t[:, None] * d
    return points


def _rays(pixels: np.ndarray, k: frames.Intrinsics) -> np.ndarray:
    """(N, 3) optical-frame rays with z = 1 through each pixel."""
    p = np.asarray(pixels, np.float64)
    return np.stack(
        [(p[:, 0] - k.cx) / k.fx, (p[:, 1] - k.cy) / k.fy, np.ones(len(p))],
        axis=1,
    )
