"""Tests for minifsd.perception.obstacles: depth image to obstacle points."""

import numpy as np
import pytest

from minifsd.geometry import frames
from minifsd.perception import obstacles
from minifsd.world import floor_plan, mjcf, sim

K = frames.intrinsics_from_fovy(320, 240, 50.0)
H = 0.5  # camera height
# The camera's pose in base_link: up H, optical axes as in R4.
ROTATION = frames.OPTICAL_TO_BASE
TRANSLATION = np.array([0.0, 0.0, H])


def _floor_depth() -> np.ndarray:
    # A level camera at height H over an empty floor: rows below the
    # horizon see the floor at depth H * fy / (v - cy); rows above see
    # nothing (no reading).
    v = np.arange(240, dtype=np.float64)[:, None].repeat(320, 1)
    depth = np.where(v > K.cy, H * K.fy / np.maximum(v - K.cy, 1e-9), 0.0)
    return depth.astype(np.float32)


def test_an_empty_floor_has_no_obstacles() -> None:
    points = obstacles.obstacle_points(_floor_depth(), K, ROTATION, TRANSLATION)
    assert len(points) == 0


def test_a_wall_ahead_is_an_obstacle_at_its_distance() -> None:
    # Every pixel sees a wall 3 m ahead (depth along the axis is 3 m).
    depth = np.full((240, 320), 3.0, dtype=np.float32)
    points = obstacles.obstacle_points(depth, K, ROTATION, TRANSLATION)
    assert len(points) > 0
    np.testing.assert_allclose(points[:, 0], 3.0, atol=1e-5)  # base x
    z = points[:, 2]
    assert z.min() >= obstacles.MIN_HEIGHT and z.max() <= obstacles.MAX_HEIGHT
    assert points[:, 1].min() < -1.0 < 1.0 < points[:, 1].max()  # spans y


def test_points_beyond_the_range_are_dropped() -> None:
    depth = np.full((240, 320), obstacles.MAX_RANGE + 1.0, dtype=np.float32)
    points = obstacles.obstacle_points(depth, K, ROTATION, TRANSLATION)
    assert len(points) == 0


@pytest.mark.gl
def test_sim_depth_puts_the_end_wall_where_the_plan_has_it() -> None:
    # End to end with the real renderer: the robot faces +x down the
    # hallway, so obstacle points straight ahead lie on the end wall.
    plan = floor_plan.generate_floor_plan(seed=0, rooms_per_side=1)
    s = sim.Sim(plan)
    k = frames.intrinsics_from_fovy(
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, mjcf.FOVY_DEG
    )
    s.set_pose(floor_plan.Pose2D(plan.length - 3.0, 0.0, 0.0))
    points = obstacles.obstacle_points(
        s.render_depth(), k, ROTATION, TRANSLATION
    )
    s.close()
    ahead = points[np.abs(points[:, 1]) < 0.3]
    np.testing.assert_allclose(ahead[:, 0], 3.0, atol=0.05)
