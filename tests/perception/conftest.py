"""Shared scenes for the perception tests: a camera over a floor and a wall.

Session-scoped fixtures that return functions, so Hypothesis tests can use
them (it rejects function-scoped fixtures, which wouldn't reset per example).
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

from minifsd.geometry import frames, ground

K = frames.intrinsics_from_fovy(320, 240, 50.0)

Pose = tuple[np.ndarray, np.ndarray]


def pose(height: float, pitch: float) -> Pose:
    """The camera in base_link: up `height`, tilted down `pitch` rad.

    base_link's z = 0 is the floor. A rotation about base y by +pitch turns
    the camera's forward axis (base x) down towards the floor.
    """
    c, s = math.cos(pitch), math.sin(pitch)
    tilt = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return tilt @ frames.OPTICAL_TO_BASE, np.array([0.0, 0.0, height])


def room(
    height: float, pitch: float, wall: float
) -> tuple[np.ndarray, np.ndarray]:
    """True depth and floor mask: a floor, and a wall `wall` metres ahead.

    The wall is the plane x = wall in base_link, across the whole view.
    """
    rotation, translation = pose(height, pitch)
    v, u = np.mgrid[0 : K.height, 0 : K.width]
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64)
    floor = ground.floor_depth(pixels, K, rotation, translation)
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


@pytest.fixture(scope="session")
def make_pose() -> Callable[[float, float], Pose]:
    """pose(), as a fixture."""
    return pose


@pytest.fixture(scope="session")
def make_room() -> Callable[
    [float, float, float], tuple[np.ndarray, np.ndarray]
]:
    """room(), as a fixture."""
    return room
