"""Tests for minifsd.world.mjcf: FloorPlan to a MuJoCo model.

Every test compiles the XML with MuJoCo itself, so "the XML looks right" is
never enough: MuJoCo has to accept it, and the compiled model has to have the
parts the simulator (R3) and the ROS node (R5) look up by name.
"""

import dataclasses

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


def test_clutter_boxes_are_solid_and_turned_by_their_yaw() -> None:
    from minifsd.world import clutter

    box = clutter.Box(3.0, 0.2, 0.7, 0.4, 0.25, 0.6, (0.4, 0.3, 0.2, 1.0))
    model = mujoco.MjModel.from_xml_string(mjcf.build_mjcf(PLAN, (box,)))
    data = _at_spawn(model)
    geom = model.geom("clutter_0")
    np.testing.assert_allclose(geom.size, [0.4, 0.25, 0.3])
    np.testing.assert_allclose(geom.pos, [3.0, 0.2, 0.3])
    # The box's own x axis points along its yaw. MuJoCo stores quaternions
    # as (w, x, y, z), so a wrong order would turn it the wrong way.
    x_axis = data.geom("clutter_0").xmat.reshape(3, 3)[:, 0]
    np.testing.assert_allclose(x_axis, [np.cos(0.7), np.sin(0.7), 0], atol=1e-9)
    assert geom.contype[0] == 1 and geom.conaffinity[0] == 1  # robot hits it


# D3: a look that differs from the default in every colour, so a test can
# tell which colour ended up where.
LOOK = mjcf.Appearance(
    floor_rgb=(0.2, 0.3, 0.6),
    floor_rgb2=(0.9, 0.8, 0.1),
    floor_period_m=0.5,
    wall_rgb=(0.3, 0.6, 0.3),
    ceiling_rgb=(0.6, 0.2, 0.2),
)


def _floor_texels(model: mujoco.MjModel) -> np.ndarray:
    """The floor texture's pixels as (n, 3) floats in [0, 1]."""
    t = model.texture(mjcf.FLOOR_TEXTURE)
    channels = int(t.nchannel[0])
    n = int(t.width[0] * t.height[0]) * channels
    texels = model.tex_data[int(t.adr[0]) : int(t.adr[0]) + n]
    return np.asarray(texels.reshape(-1, channels)[:, :3] / 255.0)


def _share_of(texels: np.ndarray, rgb: tuple[float, ...]) -> float:
    """Fraction of texels that are this colour (to MuJoCo's 8 bits)."""
    return float(np.all(np.abs(texels - rgb) < 1.5 / 255, axis=1).mean())


def test_the_default_look_keeps_the_old_colours() -> None:
    # The ROS sim and every earlier test use the default, so it must look
    # like the world did before D3: a two-tone beige checker floor, light
    # grey walls and ceiling.
    look = mjcf.Appearance()
    assert look.floor_kind == "checker"
    model = _compile(PLAN)
    texels = _floor_texels(model)
    assert _share_of(texels, (0.55, 0.5, 0.45)) == pytest.approx(0.5)
    assert _share_of(texels, (0.45, 0.4, 0.35)) == pytest.approx(0.5)
    np.testing.assert_allclose(model.geom("wall_0").rgba, (0.85, 0.85, 0.8, 1))
    np.testing.assert_allclose(
        model.geom(mjcf.CEILING).rgba, (0.92, 0.92, 0.9, 1)
    )


# Share of floor texels in the second colour, per kind. Checker is half
# and half; tiles are grout lines a texel wide at the tile edges; carpet is
# random flecks; plain has none.
SECOND_COLOUR_SHARE = {
    "checker": (0.45, 0.55),
    "tiles": (0.001, 0.05),
    "carpet": (mjcf.CARPET_FLECKS - 0.02, mjcf.CARPET_FLECKS + 0.02),
    "plain": (0.0, 0.0),
}


@pytest.mark.parametrize("kind", mjcf.FLOOR_KINDS)
def test_each_floor_kind_is_its_own_pattern(kind: str) -> None:
    assert set(SECOND_COLOUR_SHARE) == set(mjcf.FLOOR_KINDS)
    look = dataclasses.replace(LOOK, floor_kind=kind)
    model = mujoco.MjModel.from_xml_string(
        mjcf.build_mjcf(PLAN, appearance=look)
    )
    texels = _floor_texels(model)
    first = _share_of(texels, look.floor_rgb)
    second = _share_of(texels, look.floor_rgb2)
    assert first + second == pytest.approx(1.0)  # no other colours
    low, high = SECOND_COLOUR_SHARE[kind]
    assert low <= second <= high


def test_floor_pattern_repeats_every_period_in_metres() -> None:
    # texuniform makes texrepeat "repeats per metre", so a 0.5 m tile is
    # 0.5 m in every floor plan, whatever its size.
    look = dataclasses.replace(LOOK, floor_kind="tiles")
    model = mujoco.MjModel.from_xml_string(
        mjcf.build_mjcf(PLAN, appearance=look)
    )
    mat = model.material(mjcf.FLOOR_MATERIAL)
    assert mat.texuniform[0]
    np.testing.assert_allclose(mat.texrepeat, [2.0, 2.0])


def test_walls_and_ceiling_take_the_look_colours() -> None:
    look = LOOK
    model = mujoco.MjModel.from_xml_string(
        mjcf.build_mjcf(PLAN, appearance=look)
    )
    for i in range(len(PLAN.walls)):
        np.testing.assert_allclose(
            model.geom(f"wall_{i}").rgba[:3], look.wall_rgb, rtol=1e-6
        )
    np.testing.assert_allclose(
        model.geom(mjcf.CEILING).rgba[:3], look.ceiling_rgb, rtol=1e-6
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"floor_kind": "marble"},
        {"floor_rgb": (1.2, 0.5, 0.5)},
        {"wall_rgb": (0.5, -0.1, 0.5)},
        {"floor_period_m": 0.0},
    ],
)
def test_rejects_a_look_mujoco_cannot_draw(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        mjcf.Appearance(**bad)  # type: ignore[arg-type]


def _alpha(model: mujoco.MjModel, i: int) -> float:
    return float(model.geom(f"wall_{i}").rgba[3])


def _with_glass(
    share: float, plan: floor_plan.FloorPlan = PLAN
) -> mujoco.MjModel:
    look = mjcf.Appearance(glass_share=share)
    return mujoco.MjModel.from_xml_string(
        mjcf.build_mjcf(plan, appearance=look)
    )


def test_hallway_fronts_are_the_walls_facing_the_hallway() -> None:
    half = PLAN.hallway_width / 2
    fronts = mjcf.hallway_fronts(PLAN)
    assert fronts  # the plan has some
    for i, w in enumerate(PLAN.walls):
        faces_hallway = w.y_min == pytest.approx(half) or w.y_max == (
            pytest.approx(-half)
        )
        assert (i in fronts) == faces_hallway


def test_no_glass_by_default() -> None:
    model = _compile(PLAN)
    assert all(_alpha(model, i) == 1.0 for i in range(len(PLAN.walls)))


def test_full_glass_turns_only_the_hallway_fronts_to_glass() -> None:
    model = _with_glass(1.0)
    fronts = mjcf.hallway_fronts(PLAN)
    for i in range(len(PLAN.walls)):
        expected = mjcf.GLASS_RGBA[3] if i in fronts else 1.0
        assert _alpha(model, i) == pytest.approx(expected)
        # Glass still stops the robot.
        assert model.geom(f"wall_{i}").contype[0] == 1


@settings(max_examples=25, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=6),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_glass_share_sets_how_many_fronts_are_glass(
    seed: int, n: int, share: float
) -> None:
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    model = _with_glass(share, plan)
    fronts = mjcf.hallway_fronts(plan)
    glass = [i for i in fronts if _alpha(model, i) < 1.0]
    assert len(glass) == round(share * len(fronts))
    # The same plan and share pick the same segments.
    again = _with_glass(share, plan)
    assert glass == [i for i in fronts if _alpha(again, i) < 1.0]


@pytest.mark.parametrize("share", [-0.1, 1.5])
def test_rejects_a_glass_share_outside_zero_to_one(share: float) -> None:
    with pytest.raises(ValueError):
        mjcf.Appearance(glass_share=share)


def test_the_default_lighting_is_the_old_fixed_one() -> None:
    # D4 varies lighting for the data engine only; the ROS sim keeps its
    # lights: one straight down at 0.6, one angled at 0.35, no ambient.
    model = _compile(PLAN)
    np.testing.assert_allclose(model.light_diffuse, [[0.6] * 3, [0.35] * 3])
    np.testing.assert_allclose(model.light_ambient, 0.0)
    np.testing.assert_allclose(model.light_dir[0], [0, 0, -1])
    angled = np.array([1.0, 0.4, -0.6])
    np.testing.assert_allclose(
        model.light_dir[1], angled / np.linalg.norm(angled), rtol=1e-6
    )


def test_lighting_fields_reach_the_model() -> None:
    look = mjcf.Appearance(
        ambient=0.3, main_light=0.5, side_light=0.2, side_light_dir=(0, -1, -1)
    )
    model = mujoco.MjModel.from_xml_string(
        mjcf.build_mjcf(PLAN, appearance=look)
    )
    np.testing.assert_allclose(model.light_diffuse, [[0.5] * 3, [0.2] * 3])
    # Ambient light is a property of the scene, not of a direction, so it
    # goes on one light only; on both it would count twice.
    np.testing.assert_allclose(model.light_ambient, [[0.3] * 3, [0.0] * 3])
    np.testing.assert_allclose(
        model.light_dir[1], [0, -(0.5**0.5), -(0.5**0.5)], rtol=1e-6
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"ambient": 1.2},
        {"main_light": -0.1},
        {"side_light": 2.0},
        {"side_light_dir": (1.0, 0.0, 0.5)},  # pointing up, lights nothing
    ],
)
def test_rejects_lighting_mujoco_cannot_use(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        mjcf.Appearance(**bad)  # type: ignore[arg-type]
