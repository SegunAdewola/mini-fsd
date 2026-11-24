"""Builds a MuJoCo model (MJCF XML) from a FloorPlan.

The model has a floor, one box per wall, a sign plate beside each door, and
the robot. The robot is a planar base (ADR 0002): two slide joints (x, y) and
a hinge (yaw), each driven by a velocity actuator, with a forward camera on
top. The simulator (R3) turns a unicycle command (forward speed, turn rate)
into the three joint velocities.

Names that other code looks up (joints, actuators, camera, the spawn key) are
module constants, so a rename breaks the tests here, not inside ROS.
"""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import TYPE_CHECKING

from minifsd.geometry import frames
from minifsd.world import floor_plan

if TYPE_CHECKING:  # clutter imports this module, so only for type hints
    from minifsd.world import clutter as clutter_lib

SPAWN_KEY = "spawn"
JOINTS = ("base_x", "base_y", "base_yaw")
ACTUATORS = ("vel_x", "vel_y", "vel_yaw")
CAMERA = "front_camera"
CEILING = "ceiling"

CAMERA_HEIGHT = 0.5  # metres above the floor
FOVY_DEG = 50.0  # vertical field of view
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

WALL_HEIGHT = 2.5
SIGN_HEIGHT = 1.5  # centre of the sign plate
SIGN_HALF_SIZE = (0.15, 0.005, 0.1)  # half-lengths: 30 cm wide, 20 cm tall
SIGN_GAP = 0.3  # from the door's edge to the sign's centre, along the wall

ROBOT_RADIUS = 0.2
ROBOT_HALF_HEIGHT = 0.15
ROBOT_MASS = 10.0  # kg
MAX_SPEED = 3.0  # m/s, also the actuator control range
MAX_TURN = 3.0  # rad/s
# Velocity actuator gains: force = kv * (commanded - actual velocity). With
# the mass above, kv = 200 gives a time constant of m / kv = 0.05 s.
KV_LINEAR = 200.0
KV_YAW = 4.0  # yaw inertia is 0.5 * m * r^2 = 0.2 kg m^2, so also 0.05 s

WALL_RGBA = (0.85, 0.85, 0.8, 1)
CEILING_RGBA = (0.92, 0.92, 0.9, 1)
CEILING_HALF_THICKNESS = 0.01
SIGN_RGBA = (0.1, 0.3, 0.6, 1)
ROBOT_RGBA = (0.9, 0.4, 0.1, 1)


def _fmt(*values: float) -> str:
    """Space-separated numbers at full precision, as MJCF attributes want."""
    return " ".join(repr(float(v)) for v in values)


def build_mjcf(
    plan: floor_plan.FloorPlan,
    clutter: Sequence["clutter_lib.Box"] = (),
) -> str:
    """Returns MJCF XML for the floor plan with the robot at its spawn.

    Args:
        plan: The floor plan to build.
        clutter: Boxes to stand on the floor. They are solid (the robot
            collides with them) but are not part of the floor plan, so they
            stay off Nav2's map.

    Returns:
        The model as an XML string, ready for mujoco.MjModel.from_xml_string.
        Reset to the keyframe named SPAWN_KEY to place the robot.
    """
    root = ET.Element("mujoco", model=f"floor_plan_{plan.seed}")
    ET.SubElement(root, "option", timestep="0.002")
    visual = ET.SubElement(root, "visual")
    # Offscreen buffer size: the largest image the camera can render.
    ET.SubElement(
        visual,
        "global",
        offwidth=str(IMAGE_WIDTH),
        offheight=str(IMAGE_HEIGHT),
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="floor_tex",
        type="2d",
        builtin="checker",
        rgb1=".55 .5 .45",
        rgb2=".45 .4 .35",
        width="256",
        height="256",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_mat",
        texture="floor_tex",
        texrepeat="8 8",
    )

    world = ET.SubElement(root, "worldbody")
    # Directional lights with no shadows: even light in every room, like a
    # building's ceiling lights. Spot lights (MuJoCo's default) light only a
    # cone, and shadowing lights above the ceiling would leave the floor
    # dark. The second, angled light shades walls by the way they face.
    ET.SubElement(
        world,
        "light",
        directional="true",
        castshadow="false",
        pos=_fmt(plan.length / 2, 0, WALL_HEIGHT),
        dir="0 0 -1",
        diffuse=".6 .6 .6",
    )
    ET.SubElement(
        world,
        "light",
        directional="true",
        castshadow="false",
        pos=_fmt(0, 0, WALL_HEIGHT),
        dir=_fmt(1, 0.4, -0.6),
        diffuse=".35 .35 .35",
    )
    # The floor is only for looks: the robot has no vertical joint, so it
    # can't fall, and turning off its contacts avoids floor friction.
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size=_fmt(plan.length, plan.length, 0.1),
        pos=_fmt(plan.length / 2, 0, 0),
        material="floor_mat",
        contype="0",
        conaffinity="0",
    )

    for i, wall in enumerate(plan.walls):
        # MuJoCo box sizes are half-lengths, centred on pos.
        ET.SubElement(
            world,
            "geom",
            name=f"wall_{i}",
            type="box",
            size=_fmt(
                (wall.x_max - wall.x_min) / 2,
                (wall.y_max - wall.y_min) / 2,
                WALL_HEIGHT / 2,
            ),
            pos=_fmt(
                (wall.x_min + wall.x_max) / 2,
                (wall.y_min + wall.y_max) / 2,
                WALL_HEIGHT / 2,
            ),
            rgba=_fmt(*WALL_RGBA),
        )

    # Without a ceiling the camera sees empty sky above the walls once it
    # looks more than about 4 m ahead. A thin box, not a plane: planes render
    # from one side only. Visual only, like the floor.
    xs = [v for w in plan.walls for v in (w.x_min, w.x_max)]
    ys = [v for w in plan.walls for v in (w.y_min, w.y_max)]
    ET.SubElement(
        world,
        "geom",
        name=CEILING,
        type="box",
        size=_fmt(
            (max(xs) - min(xs)) / 2,
            (max(ys) - min(ys)) / 2,
            CEILING_HALF_THICKNESS,
        ),
        pos=_fmt(
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            WALL_HEIGHT + CEILING_HALF_THICKNESS,
        ),
        rgba=_fmt(*CEILING_RGBA),
        contype="0",
        conaffinity="0",
    )

    half_hall = plan.hallway_width / 2
    for room in plan.rooms:
        # Just past the door's +x edge, flat against the hallway face of the
        # wall. Visual only: no contacts.
        x = room.door_x + plan.door_width / 2 + SIGN_GAP
        y = room.side * (half_hall - SIGN_HALF_SIZE[1])
        ET.SubElement(
            world,
            "geom",
            name=f"sign_{room.number}",
            type="box",
            size=_fmt(*SIGN_HALF_SIZE),
            pos=_fmt(x, y, SIGN_HEIGHT),
            rgba=_fmt(*SIGN_RGBA),
            contype="0",
            conaffinity="0",
        )

    for i, box in enumerate(clutter):
        # MuJoCo quaternions are (w, x, y, z); ours are ROS order (x, y, z, w).
        q = frames.yaw_to_quaternion(box.yaw)
        ET.SubElement(
            world,
            "geom",
            name=f"clutter_{i}",
            type="box",
            size=_fmt(box.half_x, box.half_y, box.height / 2),
            pos=_fmt(box.x, box.y, box.height / 2),
            quat=_fmt(q.w, q.x, q.y, q.z),
            rgba=_fmt(*box.rgba),
        )

    _add_robot(world)

    actuators = ET.SubElement(root, "actuator")
    for name, joint, kv, limit in zip(
        ACTUATORS,
        JOINTS,
        (KV_LINEAR, KV_LINEAR, KV_YAW),
        (MAX_SPEED, MAX_SPEED, MAX_TURN),
        strict=True,
    ):
        ET.SubElement(
            actuators,
            "velocity",
            name=name,
            joint=joint,
            kv=_fmt(kv),
            ctrlrange=_fmt(-limit, limit),
            forcerange=_fmt(-1000, 1000),
        )

    keyframes = ET.SubElement(root, "keyframe")
    s = plan.spawn
    ET.SubElement(keyframes, "key", name=SPAWN_KEY, qpos=_fmt(s.x, s.y, s.yaw))

    return ET.tostring(root, encoding="unicode")


def _add_robot(world: ET.Element) -> None:
    """Adds the planar base with its camera to the worldbody."""
    base = ET.SubElement(world, "body", name="base", pos="0 0 0")
    # Joint positions are the pose: qpos = (x, y, yaw) in the world frame.
    x, y, yaw = JOINTS
    ET.SubElement(base, "joint", name=x, type="slide", axis="1 0 0")
    ET.SubElement(base, "joint", name=y, type="slide", axis="0 1 0")
    ET.SubElement(base, "joint", name=yaw, type="hinge", axis="0 0 1")
    ET.SubElement(
        base,
        "geom",
        name="base_geom",
        type="cylinder",
        size=_fmt(ROBOT_RADIUS, ROBOT_HALF_HEIGHT),
        pos=_fmt(0, 0, ROBOT_HALF_HEIGHT + 0.02),  # 2 cm clear of the floor
        mass=_fmt(ROBOT_MASS),
        rgba=_fmt(*ROBOT_RGBA),
    )
    # MuJoCo cameras look along their own -z axis, with +y up in the image.
    # xyaxes gives the camera's x and y axes in the body frame: x = -y_body
    # (image right is the robot's right) and y = +z_body (image up is up).
    # Then -z = x cross y, negated, = +x_body: the camera looks forward.
    ET.SubElement(
        base,
        "camera",
        name=CAMERA,
        pos=_fmt(0, 0, CAMERA_HEIGHT),
        xyaxes="0 -1 0 0 0 1",
        fovy=_fmt(FOVY_DEG),
    )
