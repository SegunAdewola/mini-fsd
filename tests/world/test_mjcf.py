"""Tests for minifsd.world.mjcf: FloorPlan to a MuJoCo model.

Every test compiles the XML with MuJoCo itself, so "the XML looks right" is
never enough: MuJoCo has to accept it, and the compiled model has to have the
parts the simulator (R3) and the ROS node (R5) look up by name.
"""

import mujoco
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.world import floor_plan, mjcf

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=2)


def _compile(plan: floor_plan.FloorPlan) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(mjcf.build_mjcf(plan))


def _at_spawn(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(mjcf.SPAWN_KEY).id)
    mujoco.mj_forward(model, data)
    return data


def test_compiles() -> None:
    _compile(PLAN)


def test_robot_parts_exist_by_name() -> None:
    model = _compile(PLAN)
    for name in mjcf.JOINTS:
        model.joint(name)  # raises KeyError if missing
    for name in mjcf.ACTUATORS:
        model.actuator(name)
    model.camera(mjcf.CAMERA)


def test_base_moves_in_the_plane_only() -> None:
    model = _compile(PLAN)
    x, y, yaw = (model.joint(n) for n in mjcf.JOINTS)
    assert x.type[0] == mujoco.mjtJoint.mjJNT_SLIDE
    assert y.type[0] == mujoco.mjtJoint.mjJNT_SLIDE
    assert yaw.type[0] == mujoco.mjtJoint.mjJNT_HINGE
    np.testing.assert_allclose(x.axis, [1, 0, 0])
    np.testing.assert_allclose(y.axis, [0, 1, 0])
    np.testing.assert_allclose(yaw.axis, [0, 0, 1])
    assert model.nq == 3  # nothing else can move


def test_spawn_key_puts_robot_at_spawn_pose() -> None:
    model = _compile(PLAN)
    data = _at_spawn(model)
    s = PLAN.spawn
    np.testing.assert_allclose(data.qpos, [s.x, s.y, s.yaw])


def test_camera_is_half_a_metre_up_and_looks_along_plus_x() -> None:
    model = _compile(PLAN)
    data = _at_spawn(model)
    cam = data.camera(mjcf.CAMERA)
    np.testing.assert_allclose(
        cam.xpos, [PLAN.spawn.x, PLAN.spawn.y, mjcf.CAMERA_HEIGHT], atol=1e-9
    )
    # A MuJoCo camera looks along its own -z axis: the third column of its
    # rotation matrix, negated.
    forward = -cam.xmat.reshape(3, 3)[:, 2]
    np.testing.assert_allclose(forward, [1, 0, 0], atol=1e-9)
    up = cam.xmat.reshape(3, 3)[:, 1]
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-9)


def test_camera_resolution_and_field_of_view() -> None:
    model = _compile(PLAN)
    assert model.camera(mjcf.CAMERA).fovy[0] == pytest.approx(mjcf.FOVY_DEG)
    assert (model.vis.global_.offwidth, model.vis.global_.offheight) == (
        mjcf.IMAGE_WIDTH,
        mjcf.IMAGE_HEIGHT,
    )


def test_one_box_per_wall_with_matching_footprint() -> None:
    model = _compile(PLAN)
    for i, wall in enumerate(PLAN.walls):
        geom = model.geom(f"wall_{i}")
        assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
        half = [
            (wall.x_max - wall.x_min) / 2,
            (wall.y_max - wall.y_min) / 2,
            mjcf.WALL_HEIGHT / 2,
        ]
        centre = [
            (wall.x_min + wall.x_max) / 2,
            (wall.y_min + wall.y_max) / 2,
            mjcf.WALL_HEIGHT / 2,
        ]
        np.testing.assert_allclose(geom.size, half)
        np.testing.assert_allclose(geom.pos, centre)


def test_one_sign_per_room_on_the_hallway_side_of_its_door() -> None:
    model = _compile(PLAN)
    half_hall = PLAN.hallway_width / 2
    for room in PLAN.rooms:
        pos = model.geom(f"sign_{room.number}").pos
        # Beside the door, not in it: more than half a door from its centre.
        assert abs(pos[0] - room.door_x) > PLAN.door_width / 2
        # On this room's side, just inside the hallway, at eye height.
        assert np.sign(pos[1]) == room.side
        assert half_hall - 0.05 < abs(pos[1]) < half_hall
        assert pos[2] == pytest.approx(mjcf.SIGN_HEIGHT)


@settings(max_examples=25, deadline=None)  # compiling takes a few ms each
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=6),
)
def test_any_floor_plan_compiles(seed: int, n: int) -> None:
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    model = _compile(plan)
    walls = sum(
        model.geom(i).name.startswith("wall_") for i in range(model.ngeom)
    )
    assert walls == len(plan.walls)


def test_ceiling_covers_the_floor_plan_at_wall_height() -> None:
    model = _compile(PLAN)
    ceiling = model.geom(mjcf.CEILING)
    # A thin box (planes render from one side only); its underside is at
    # the top of the walls.
    assert ceiling.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert ceiling.pos[2] - ceiling.size[2] == pytest.approx(mjcf.WALL_HEIGHT)
    xs = [v for w in PLAN.walls for v in (w.x_min, w.x_max)]
    ys = [v for w in PLAN.walls for v in (w.y_min, w.y_max)]
    # Box sizes are half-lengths, centred on pos.
    eps = 1e-9  # float noise from centre +- half-size
    assert ceiling.pos[0] - ceiling.size[0] <= min(xs) + eps
    assert ceiling.pos[0] + ceiling.size[0] >= max(xs) - eps
    assert ceiling.pos[1] - ceiling.size[1] <= min(ys) + eps
    assert ceiling.pos[1] + ceiling.size[1] >= max(ys) - eps
    # Visual only: the robot can't reach it anyway.
    assert ceiling.contype[0] == 0 and ceiling.conaffinity[0] == 0
