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

from minifsd.geometry import camera, frames
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


@pytest.mark.gl
def test_rgb_image_shape(s: sim.Sim) -> None:
    rgb = s.render_rgb()
    assert rgb.shape == (mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH, 3)
    assert rgb.dtype == np.uint8
    assert rgb.std() > 0  # not a blank image


@pytest.mark.gl
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


def test_set_pose_moves_the_robot_without_stepping(s: sim.Sim) -> None:
    target = floor_plan.Pose2D(3.0, 0.4, 1.2)
    s.set_pose(target)
    p = s.pose()
    assert (p.x, p.y, p.yaw) == pytest.approx((3.0, 0.4, 1.2))
    assert s.time == 0.0
    assert s.velocity() == pytest.approx((0.0, 0.0))


@pytest.mark.gl
def test_floor_mask_is_the_bottom_of_the_image(s: sim.Sim) -> None:
    mask = s.render_floor_mask()
    assert mask.shape == (mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH)
    assert mask.dtype == np.bool_
    # A level camera sees floor below the horizon and none above it.
    assert mask[-1].all()
    assert not mask[: mjcf.IMAGE_HEIGHT // 2 - 2].any()


@pytest.mark.gl
def test_floor_depth_matches_the_ground_plane(s: sim.Sim) -> None:
    # For a level camera at height h, a floor pixel in row v is at depth
    # h * fy / (v - cy) along the viewing axis. This ties the floor mask,
    # the depth image and the R4 intrinsics together.
    k = frames.intrinsics_from_fovy(
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, mjcf.FOVY_DEG
    )
    mask, depth = s.render_floor_mask(), s.render_depth()
    rows = np.arange(mjcf.IMAGE_HEIGHT)[:, None].repeat(mjcf.IMAGE_WIDTH, 1)
    near = mask & (rows > k.cy + 20)  # well below the horizon
    expected = mjcf.CAMERA_HEIGHT * k.fy / (rows[near] - k.cy)
    np.testing.assert_allclose(depth[near], expected, rtol=0.02)


@pytest.mark.gl
def test_no_sky_down_a_long_hallway(s: sim.Sim) -> None:
    # Looking along the hallway, every pixel hits geometry inside the
    # building, so no depth is beyond its far corner.
    xs = [v for w in PLAN.walls for v in (w.x_min, w.x_max)]
    ys = [v for w in PLAN.walls for v in (w.y_min, w.y_max)]
    diagonal = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    s.set_pose(floor_plan.Pose2D(0.5, 0.0, 0.0))
    depth = s.render_depth()
    assert depth.max() < diagonal + mjcf.WALL_HEIGHT


@pytest.mark.gl
def test_floor_is_lit_under_the_ceiling(s: sim.Sim) -> None:
    # Lights above the ceiling would leave the floor in its shadow. The
    # floor texture's colours are about 100 to 140 of 255; lit, the floor
    # pixels should average well above black.
    rgb, mask = s.render_rgb(), s.render_floor_mask()
    assert rgb[mask].mean() > 60


def test_clutter_blocks_the_robot() -> None:
    from minifsd.world import clutter

    box = clutter.Box(4.0, 0.0, 0.0, 0.3, 0.3, 0.6, (0.4, 0.3, 0.2, 1.0))
    s = sim.Sim(PLAN, clutter=(box,))
    _drive(s, 1.0, 0.0, 6.0)
    assert s.pose().x < box.x - box.half_x - mjcf.ROBOT_RADIUS + 0.02


def _rot_x(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def _optical_to_world(
    pose: floor_plan.Pose2D, cam: sim.CameraSetup
) -> np.ndarray:
    """World-from-optical rotation, built here independently of sim.py.

    A level camera facing +x has optical x (image right) = world -y,
    optical y (image down) = world -z and optical z (ahead) = world +x.
    Pitching down turns the view axis toward image-down, a rotation about
    optical x by -pitch. A positive roll lowers the image's right side, a
    rotation about optical z by +roll.
    """
    level = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    r = _rot_z(pose.yaw) @ level @ _rot_x(-cam.pitch) @ _rot_z(cam.roll)
    return np.asarray(r, dtype=float)


@pytest.mark.gl
@pytest.mark.parametrize(
    ("height", "pitch_deg", "roll_deg"),
    [
        (0.3, 0.0, 0.0),
        (0.5, 25.0, 0.0),
        (1.0, 25.0, -5.0),
        (1.6, 10.0, 5.0),
        (1.6, 0.0, 0.0),
    ],
)
def test_floor_pixels_land_on_the_floor_for_any_camera_pose(
    s: sim.Sim, height: float, pitch_deg: float, roll_deg: float
) -> None:
    # D1: the camera can sit anywhere from robot height to a handheld phone,
    # tilted down and slightly rolled. Every floor pixel, lifted through its
    # depth and the camera pose, must lie on the floor plane z = 0. A wrong
    # sign or rotation order puts these points metres off.
    pose = floor_plan.Pose2D(PLAN.spawn.x, PLAN.spawn.y, 0.3)
    cam = sim.CameraSetup(
        height, math.radians(pitch_deg), math.radians(roll_deg)
    )
    s.set_pose(pose)
    s.set_camera(cam)
    mask, depth = s.render_floor_mask(), s.render_depth()
    k = frames.intrinsics_from_fovy(
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, mjcf.FOVY_DEG
    )
    depth = np.where(mask & (depth < 6.0), depth, 0.0)  # floor, not too far
    points, _ = camera.back_project(depth, k, stride=4)
    assert len(points) > 100  # it does see floor
    world = points @ _optical_to_world(pose, cam).T
    world += (pose.x, pose.y, height)
    # Depth comes from a float depth buffer; 3 cm allows for it at range.
    assert np.percentile(np.abs(world[:, 2]), 99) < 0.03


@pytest.mark.gl
def test_the_camera_does_not_see_its_own_robot(s: sim.Sim) -> None:
    # Tilted down at robot height, the bottom of the view passes over the
    # robot's top (0.32 m up, 0.2 m radius). The camera should see floor
    # there, not the body it is mounted on.
    s.set_camera(sim.CameraSetup(0.5, math.radians(25.0), 0.0))
    assert s.render_floor_mask()[-1].all()


@pytest.mark.gl
def test_the_default_camera_pose_restores_the_robot_camera(s: sim.Sim) -> None:
    before = s.render_depth()
    s.set_camera(sim.CameraSetup(1.2, 0.3, 0.05))
    s.set_camera(sim.CameraSetup())
    np.testing.assert_allclose(s.render_depth(), before, atol=1e-4)
    assert sim.CameraSetup().height == mjcf.CAMERA_HEIGHT


@pytest.mark.gl
@pytest.mark.parametrize("fovy_deg", [50.0, 62.0, 75.0])
def test_reported_intrinsics_match_what_a_lens_renders(
    s: sim.Sim, fovy_deg: float
) -> None:
    # D2: the field of view varies, and the intrinsics the sim reports must
    # be the ones it rendered with. Back-projecting floor pixels through the
    # reported intrinsics lands them on z = 0; with the wrong fy the rays
    # spread too much or too little and the floor bends up or down.
    pose = floor_plan.Pose2D(PLAN.spawn.x, PLAN.spawn.y, 0.0)
    cam = sim.CameraSetup(1.0, math.radians(15.0), 0.0, fovy_deg)
    s.set_pose(pose)
    s.set_camera(cam)
    k = s.intrinsics()
    assert k == frames.intrinsics_from_fovy(
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, fovy_deg
    )
    mask, depth = s.render_floor_mask(), s.render_depth()
    depth = np.where(mask & (depth < 6.0), depth, 0.0)
    points, _ = camera.back_project(depth, k, stride=4)
    assert len(points) > 100
    world = points @ _optical_to_world(pose, cam).T
    world += (pose.x, pose.y, cam.height)
    assert np.percentile(np.abs(world[:, 2]), 99) < 0.03


def test_the_default_setup_is_the_robot_camera() -> None:
    assert sim.CameraSetup().fovy_deg == mjcf.FOVY_DEG


# D3 looks: every floor kind, in colours far from the default.
LOOKS = [
    mjcf.Appearance(
        floor_kind=kind,
        floor_rgb=(0.2, 0.3, 0.6),
        floor_rgb2=(0.9, 0.8, 0.1),
        floor_period_m=0.5,
        wall_rgb=(0.3, 0.6, 0.3),
        ceiling_rgb=(0.6, 0.2, 0.2),
    )
    for kind in mjcf.FLOOR_KINDS
]


@pytest.mark.gl
@pytest.mark.parametrize("look", LOOKS, ids=lambda a: a.floor_kind)
def test_the_floor_label_does_not_depend_on_the_look(
    s: sim.Sim, look: mjcf.Appearance
) -> None:
    # Labels come from geom ids, not colours, so a new floor must change the
    # picture and leave the floor mask exactly as it was.
    other = sim.Sim(PLAN, appearance=look)
    try:
        for each in (s, other):
            each.set_camera(sim.CameraSetup(1.2, math.radians(20.0), 0.0))
        np.testing.assert_array_equal(
            other.render_floor_mask(), s.render_floor_mask()
        )
        assert not np.array_equal(other.render_rgb(), s.render_rgb())
    finally:
        other.close()


@pytest.mark.gl
def test_the_same_look_renders_the_same_pixels() -> None:
    a = sim.Sim(PLAN, appearance=LOOKS[2])
    b = sim.Sim(PLAN, appearance=LOOKS[2])
    try:
        np.testing.assert_array_equal(a.render_rgb(), b.render_rgb())
    finally:
        a.close()
        b.close()


@pytest.mark.gl
def test_glass_is_labelled_as_the_wall_it_is(s: sim.Sim) -> None:
    # D3b: through glass the camera sees the room behind, but the robot
    # can't drive there. So the labels must be those of a solid wall: the
    # floor mask stops at the glass and depth reads the glass. Only the
    # picture changes.
    glass = sim.Sim(PLAN, appearance=mjcf.Appearance(glass_share=1.0))
    try:
        # Facing a hallway front, side-on, from the middle of the hallway.
        pose = floor_plan.Pose2D(PLAN.length / 2 + 0.4, 0.0, math.pi / 2)
        for each in (s, glass):
            each.set_pose(pose)
            each.set_camera(sim.CameraSetup(0.5, math.radians(10.0), 0.0))
        np.testing.assert_array_equal(
            glass.render_floor_mask(), s.render_floor_mask()
        )
        np.testing.assert_allclose(
            glass.render_depth(), s.render_depth(), atol=1e-4
        )
        assert not np.array_equal(glass.render_rgb(), s.render_rgb())
    finally:
        glass.close()
