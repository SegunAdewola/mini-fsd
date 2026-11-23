"""Coordinate frames: quaternions, the camera optical frame and intrinsics.

ROS has fixed conventions that MuJoCo doesn't share, and this module holds the
conversions between them, so the ROS node (R5) only copies numbers into
messages.

- REP 103: the base frame is x forward, y left, z up. Angles are
  counterclockwise when seen from above.
- Quaternions are (x, y, z, w), the order geometry_msgs uses. MuJoCo stores
  (w, x, y, z), so never pass one straight to the other.
- A camera's optical frame is x right, y down, z forward (into the scene).
  Images, CameraInfo and depth point clouds are in this frame.
"""

import dataclasses
import math
from typing import NamedTuple

import numpy as np


class Quaternion(NamedTuple):
    """A rotation as a unit quaternion, in ROS order."""

    x: float
    y: float
    z: float
    w: float


def yaw_to_quaternion(yaw: float) -> Quaternion:
    """The rotation by yaw radians about +z.

    Args:
        yaw: Heading, radians, counterclockwise.

    Returns:
        The unit quaternion for that rotation.
    """
    # A rotation by angle a about a unit axis n is (n sin(a/2), cos(a/2)).
    return Quaternion(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def quaternion_to_yaw(q: Quaternion) -> float:
    """The heading of a rotation, in [-pi, pi].

    Args:
        q: A unit quaternion. Any roll and pitch in it is ignored.

    Returns:
        The angle of the rotated x axis, measured from +x in the xy plane.
    """
    # The rotated x axis is the first column of the rotation matrix; its x
    # and y entries are these two terms.
    return math.atan2(
        2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)
    )


def quaternion_from_matrix(m: np.ndarray) -> Quaternion:
    """Converts a 3x3 rotation matrix to a quaternion.

    Uses Shepperd's method: it divides by the largest of four candidate
    terms, so it stays accurate for every rotation, including half turns
    where w is near zero.

    Args:
        m: A proper rotation matrix (orthonormal, determinant +1).

    Returns:
        The quaternion for the same rotation, with w >= 0.
    """
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    candidates = (trace, m[0, 0], m[1, 1], m[2, 2])
    biggest = int(np.argmax(candidates))
    if biggest == 0:
        s = 2 * math.sqrt(1 + trace)  # s = 4w
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
        w = s / 4
    elif biggest == 1:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])  # s = 4x
        x = s / 4
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
        w = (m[2, 1] - m[1, 2]) / s
    elif biggest == 2:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])  # s = 4y
        x = (m[0, 1] + m[1, 0]) / s
        y = s / 4
        z = (m[1, 2] + m[2, 1]) / s
        w = (m[0, 2] - m[2, 0]) / s
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])  # s = 4z
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = s / 4
        w = (m[1, 0] - m[0, 1]) / s
    if w < 0:  # q and -q are the same rotation; pick one
        x, y, z, w = -x, -y, -z, -w
    return Quaternion(float(x), float(y), float(z), float(w))


# Rotation from the camera's optical frame to the camera's body frame (which
# has the base's axes: x forward, y left, z up). Column i is optical axis i
# written in body axes: optical x (right) = -y, optical y (down) = -z,
# optical z (forward) = +x.
OPTICAL_TO_BASE = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
)
OPTICAL_TO_BASE_QUAT = quaternion_from_matrix(OPTICAL_TO_BASE)


@dataclasses.dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera's intrinsics, in pixels.

    A point (X, Y, Z) in the optical frame lands on pixel
    (u, v) = (fx X / Z + cx, fy Y / Z + cy), with pixel centres at whole
    numbers (the OpenCV and ROS convention).
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        """The 3x3 camera matrix K. CameraInfo.k is this, row by row."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        )


def intrinsics_from_fovy(
    width: int, height: int, fovy_deg: float
) -> Intrinsics:
    """Intrinsics of an ideal camera, like MuJoCo's, from its vertical FOV.

    MuJoCo renders with square pixels and the optical axis through the image
    centre, and has no lens distortion, so these intrinsics are exact for the
    simulated camera.

    Args:
        width: Image width, pixels.
        height: Image height, pixels.
        fovy_deg: Vertical field of view, degrees, in (0, 180).

    Returns:
        The intrinsics.

    Raises:
        ValueError: If the field of view is outside (0, 180) degrees.
    """
    if not 0 < fovy_deg < 180:
        raise ValueError(f"fovy_deg must be in (0, 180), got {fovy_deg}")
    # Half the image height spans half the field of view:
    # tan(fovy / 2) = (height / 2) / fy.
    f = (height / 2) / math.tan(math.radians(fovy_deg) / 2)
    return Intrinsics(
        width=width,
        height=height,
        fx=f,
        fy=f,
        cx=(width - 1) / 2,
        cy=(height - 1) / 2,
    )
