"""Tests for minifsd.world.sim: driving, the watchdog, limits and the camera.

Driving tests send a command every 0.05 s (20 Hz), the way Nav2's controller
and the ROS node will. Tolerances allow for the actuators' 0.05 s time
constant: a 0.5 m/s command covers about 0.475 m in its first second.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.world import floor_plan, mjcf, sim

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=2)
TICK = 0.05  # seconds between commands


def _drive(s: sim.Sim, v: float, w: float, seconds: float) -> None:
    """Sends (v, w) every TICK for the given time."""
    for _ in range(round(seconds / TICK)):
        s.command(v, w)
        s.step(TICK)


@pytest.fixture
def s() -> sim.Sim:
    return sim.Sim(PLAN)


def test_starts_at_spawn(s: sim.Sim) -> None:
    p = s.pose()
    assert (p.x, p.y, p.yaw) == pytest.approx(
        (PLAN.spawn.x, PLAN.spawn.y, PLAN.spawn.yaw)
    )
    assert s.time == 0.0


def test_step_advances_time(s: sim.Sim) -> None:
    s.step(0.25)
    assert s.time == pytest.approx(0.25)


def test_drives_half_a_metre_in_one_second(s: sim.Sim) -> None:
    _drive(s, 0.5, 0.0, 1.0)
    p = s.pose()
    assert p.x - PLAN.spawn.x == pytest.approx(0.5, abs=0.04)
    assert p.y == pytest.approx(PLAN.spawn.y, abs=1e-3)


def test_turns_one_radian_in_one_second(s: sim.Sim) -> None:
    _drive(s, 0.0, 1.0, 1.0)
    p = s.pose()
    assert p.yaw - PLAN.spawn.yaw == pytest.approx(1.0, abs=0.06)
    assert (p.x, p.y) == pytest.approx((PLAN.spawn.x, PLAN.spawn.y), abs=1e-3)


def test_drives_along_its_heading(s: sim.Sim) -> None:
    _drive(s, 0.0, math.pi / 2, 1.0)  # turn to face roughly +y
    _drive(s, 0.0, 0.0, 0.3)  # let the turn settle before reading the yaw
    yaw = s.pose().yaw
    start = s.pose()
    _drive(s, 0.5, 0.0, 1.0)
    end = s.pose()
    heading = math.atan2(end.y - start.y, end.x - start.x)
    assert heading == pytest.approx(yaw, abs=0.02)


def test_reports_body_frame_velocity(s: sim.Sim) -> None:
    _drive(s, 0.0, math.pi / 2, 1.0)  # facing sideways in the world
    _drive(s, 0.4, 0.2, 1.0)
    v, w = s.velocity()
    assert (v, w) == pytest.approx((0.4, 0.2), abs=0.01)


def test_watchdog_stops_the_robot_after_commands_stop(s: sim.Sim) -> None:
    _drive(s, 1.0, 0.5, 1.0)
    s.step(0.5)  # no commands: as if teleop crashed
    v, w = s.velocity()
    assert abs(v) < 0.01 and abs(w) < 0.01


def test_watchdog_does_not_fire_while_commands_keep_coming(s: sim.Sim) -> None:
    _drive(s, 1.0, 0.0, 2.0)
    v, _ = s.velocity()
    assert v == pytest.approx(1.0, abs=0.01)


def test_clamps_speed_and_turn_rate(s: sim.Sim) -> None:
    _drive(s, 10.0, 0.0, 0.5)
    v, _ = s.velocity()
    assert v == pytest.approx(mjcf.MAX_SPEED, abs=0.05)
    s2 = sim.Sim(PLAN)
    _drive(s2, 0.0, -10.0, 0.5)
    _, w = s2.velocity()
    assert w == pytest.approx(-mjcf.MAX_TURN, abs=0.05)


def test_walls_block_it(s: sim.Sim) -> None:
    _drive(s, 1.0, 0.0, PLAN.length + 2)  # long enough to reach the end wall
    assert s.pose().x < PLAN.length - mjcf.ROBOT_RADIUS + 0.02


def test_rgb_image_shape(s: sim.Sim) -> None:
    rgb = s.render_rgb()
    assert rgb.shape == (mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH, 3)
    assert rgb.dtype == np.uint8
    assert rgb.std() > 0  # not a blank image


def test_depth_to_the_end_wall(s: sim.Sim) -> None:
    depth = s.render_depth()
    assert depth.shape == (mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH)
    assert depth.dtype == np.float32
    centre = depth[mjcf.IMAGE_HEIGHT // 2, mjcf.IMAGE_WIDTH // 2]
    # The camera sits over the robot's centre and looks along +x, straight
    # at the end wall's inner face at x = length.
    assert centre == pytest.approx(PLAN.length - PLAN.spawn.x, abs=0.05)


@given(
    st.floats(-3, 3),
    st.floats(-3, 3),
    st.floats(-math.pi, math.pi),
)
def test_body_to_world_keeps_speed_and_turn_rate(
    v: float, w: float, yaw: float
) -> None:
    vx, vy, wz = sim.body_to_world(v, w, yaw)
    assert math.hypot(vx, vy) == pytest.approx(abs(v), abs=1e-9)
    assert wz == w
    if abs(v) > 1e-6:  # moving: along the heading, or straight back
        along = vx * math.cos(yaw) + vy * math.sin(yaw)
        assert along == pytest.approx(v, abs=1e-9)
