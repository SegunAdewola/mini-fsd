"""Builds a MuJoCo model (MJCF XML) from a FloorPlan.

The model has a floor, one box per wall, a sign plate beside each door, and
the robot. How the floor, walls and ceiling look is an Appearance (D3); the
default is the fixed look the ROS sim uses.

The robot is a planar base (ADR 0002): two slide joints (x, y) and a hinge
(yaw), each driven by a velocity actuator, with a forward camera on top. The
simulator (R3) turns a unicycle command (forward speed, turn rate) into the
three joint velocities.

Names that other code looks up (joints, actuators, camera, the spawn key) are
module constants, so a rename breaks the tests here, not inside ROS.
"""

import dataclasses
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from minifsd.geometry import frames
from minifsd.world import floor_plan, signs

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
# MuJoCo renders geom groups 0 to 2 by default. The robot's body is in group
# 3, so its own camera never sees it (D1: a camera tilted down at robot
# height would). Physics ignores groups, so it still collides.
ROBOT_GEOM_GROUP = 3
ROBOT_MASS = 10.0  # kg
MAX_SPEED = 3.0  # m/s, also the actuator control range
MAX_TURN = 3.0  # rad/s
# Velocity actuator gains: force = kv * (commanded - actual velocity). With
# the mass above, kv = 200 gives a time constant of m / kv = 0.05 s.
KV_LINEAR = 200.0
KV_YAW = 4.0  # yaw inertia is 0.5 * m * r^2 = 0.2 kg m^2, so also 0.05 s

CEILING_HALF_THICKNESS = 0.01
ROBOT_RGBA = (0.9, 0.4, 0.1, 1)

FLOOR_TEXTURE = "floor_tex"
FLOOR_MATERIAL = "floor_mat"
# D3 floor patterns, all MuJoCo built-in textures, so no image files:
#   checker  two-colour squares, two each way per period
#   tiles    one colour with grout lines in the second, one tile per period
#   carpet   one colour flecked with the second
#   plain    one colour (vinyl, polished concrete)
FLOOR_KINDS = ("checker", "tiles", "carpet", "plain")
CARPET_FLECKS = 0.3  # share of carpet texels in the second colour
TEXTURE_SIZE = 256  # texels per side of one period
# D3b glass room fronts: a faint blue-green tint, mostly see-through.
GLASS_RGBA = (0.6, 0.75, 0.75, 0.25)

RGB = tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Appearance:
    """How the floor, walls and ceiling of one floor plan look (D3).

    The default is the fixed look from before D3, which the ROS sim keeps.
    The data engine samples a new one per floor plan, so models can't learn
    "floor is a beige checker".

    Attributes:
        floor_kind: One of FLOOR_KINDS.
        floor_rgb: Main floor colour, each channel in [0, 1].
        floor_rgb2: Second checker colour, grout or flecks; unused by plain.
        floor_period_m: Metres per repeat of the floor pattern.
        wall_rgb: Wall colour.
        ceiling_rgb: Ceiling colour.
        glass_share: Share of the hallway fronts (hallway_fronts) that are
            glass, in [0, 1]. Glass still blocks the robot; only its
            colour changes.
        ambient: Ambient light in [0, 1], which lights every surface the
            same whichever way it faces (D4).
        main_light: Brightness of the light shining straight down.
        side_light: Brightness of the angled light, which shades walls by
            the way they face.
        side_light_dir: Direction the angled light shines; must point
            down (z < 0). Need not be unit length.
        sign_rgb: Colour of the sign plates (D7).
        sign_text_rgb: Colour of the room numbers on them.
        sign_font: A font name from signs.FONTS.
        sign_word: A word from signs.WORDS above the number, or "" for none.
    """

    floor_kind: str = "checker"
    floor_rgb: RGB = (0.55, 0.5, 0.45)
    floor_rgb2: RGB = (0.45, 0.4, 0.35)
    floor_period_m: float = 1.7  # about the old fixed tiling on a small plan
    wall_rgb: RGB = (0.85, 0.85, 0.8)
    ceiling_rgb: RGB = (0.92, 0.92, 0.9)
    glass_share: float = 0.0
    ambient: float = 0.0
    main_light: float = 0.6
    side_light: float = 0.35
    side_light_dir: tuple[float, float, float] = (1.0, 0.4, -0.6)
    sign_rgb: RGB = (0.1, 0.3, 0.6)
    sign_text_rgb: RGB = (1.0, 1.0, 1.0)
    sign_font: str = "simplex"
    sign_word: str = ""

    def __post_init__(self) -> None:
        """Rejects looks MuJoCo can't draw.

        Raises:
            ValueError: For an unknown kind, a colour outside [0, 1] or a
                period that isn't positive.
        """
        if self.floor_kind not in FLOOR_KINDS:
            raise ValueError(
                f"floor_kind must be one of {FLOOR_KINDS}, "
                f"got {self.floor_kind!r}"
            )
        if self.sign_font not in signs.FONTS:
            raise ValueError(
                f"sign_font must be one of {tuple(signs.FONTS)}, "
                f"got {self.sign_font!r}"
            )
        if self.sign_word not in ("", *signs.WORDS):
            raise ValueError(
                f"sign_word must be '' or one of {signs.WORDS}, "
                f"got {self.sign_word!r}"
            )
        for name in (
            "floor_rgb",
            "floor_rgb2",
            "wall_rgb",
            "ceiling_rgb",
            "sign_rgb",
            "sign_text_rgb",
        ):
            rgb = getattr(self, name)
            if len(rgb) != 3 or not all(0.0 <= c <= 1.0 for c in rgb):
                raise ValueError(f"{name} needs 3 values in [0, 1], got {rgb}")
        for name in ("ambient", "main_light", "side_light"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if len(self.side_light_dir) != 3 or self.side_light_dir[2] >= 0:
            raise ValueError(
                f"side_light_dir must point down, got {self.side_light_dir}"
            )
        if not 0.0 <= self.glass_share <= 1.0:
            raise ValueError(
                f"glass_share must be in [0, 1], got {self.glass_share}"
            )
        if self.floor_period_m <= 0:
            raise ValueError(
                f"floor_period_m must be positive, got {self.floor_period_m}"
            )


# The look from before D3. One shared instance, used as the default
# argument (Appearance is frozen, so sharing it is safe).
DEFAULT_APPEARANCE = Appearance()


def _fmt(*values: float) -> str:
    """Space-separated numbers at full precision, as MJCF attributes want."""
    return " ".join(repr(float(v)) for v in values)


def build_mjcf(
    plan: floor_plan.FloorPlan,
    clutter: Sequence["clutter_lib.Box"] = (),
    appearance: Appearance = DEFAULT_APPEARANCE,
) -> str:
    """Returns MJCF XML for the floor plan with the robot at its spawn.

    Args:
        plan: The floor plan to build.
        clutter: Boxes to stand on the floor. They are solid (the robot
            collides with them) but are not part of the floor plan, so they
            stay off Nav2's map.
        appearance: Colours and floor pattern.

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
    _add_floor_texture(asset, appearance)
    for room in plan.rooms:
        # A blank texture of the right size; signs.paint draws the number
        # into it once the model is compiled.
        ET.SubElement(
            asset,
            "texture",
            name=signs.texture_name(room.number),
            type="2d",
            builtin="flat",
            rgb1=_fmt(*appearance.sign_rgb),
            width=str(signs.TEXTURE_WIDTH),
            height=str(signs.TEXTURE_HEIGHT),
        )
        ET.SubElement(
            asset,
            "material",
            name=signs.material_name(room.number),
            texture=signs.texture_name(room.number),
        )

    world = ET.SubElement(root, "worldbody")
    # Directional lights with no shadows: even light in every room, like a
    # building's ceiling lights. Spot lights (MuJoCo's default) light only a
    # cone, and shadowing lights above the ceiling would leave the floor
    # dark. The second, angled light shades walls by the way they face.
    # Ambient goes on the first light only: MuJoCo adds up every light's
    # ambient, so on both it would count twice.
    look = appearance
    ET.SubElement(
        world,
        "light",
        directional="true",
        castshadow="false",
        pos=_fmt(plan.length / 2, 0, WALL_HEIGHT),
        dir="0 0 -1",
        diffuse=_fmt(*[look.main_light] * 3),
        ambient=_fmt(*[look.ambient] * 3),
    )
    ET.SubElement(
        world,
        "light",
        directional="true",
        castshadow="false",
        pos=_fmt(0, 0, WALL_HEIGHT),
        dir=_fmt(*look.side_light_dir),
        diffuse=_fmt(*[look.side_light] * 3),
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
        material=FLOOR_MATERIAL,
        contype="0",
        conaffinity="0",
    )

    glass = _glass_walls(plan, appearance.glass_share)
    for i, wall in enumerate(plan.walls):
        rgba = GLASS_RGBA if i in glass else (*appearance.wall_rgb, 1)
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
            rgba=_fmt(*rgba),
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
        rgba=_fmt(*appearance.ceiling_rgb, 1),
        contype="0",
        conaffinity="0",
    )

    half_hall = plan.hallway_width / 2
    for room in plan.rooms:
        # Just past the door's +x edge, flat against the hallway face of the
        # wall. Visual only: no contacts.
        x = room.door_x + plan.door_width / 2 + SIGN_GAP
        y = room.side * (half_hall - SIGN_HALF_SIZE[1])
        # MuJoCo maps a texture onto a box face in one fixed direction, so
        # the text reads correctly from -y and mirrored from +y. Signs on the
        # right wall, seen from +y, are turned half round about z.
        quat = "1 0 0 0" if room.side == floor_plan.LEFT else "0 0 0 1"
        ET.SubElement(
            world,
            "geom",
            name=f"sign_{room.number}",
            type="box",
            size=_fmt(*SIGN_HALF_SIZE),
            pos=_fmt(x, y, SIGN_HEIGHT),
            quat=quat,
            material=signs.material_name(room.number),
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


def hallway_fronts(plan: floor_plan.FloorPlan) -> list[int]:
    """Indices into plan.walls of the wall segments facing the hallway.

    These are the room fronts between doors, the walls that are glass in
    many campus buildings. End walls (which span both sides) and the rooms'
    back and side walls are not included.
    """
    half = plan.hallway_width / 2
    eps = 1e-9
    return [
        i
        for i, w in enumerate(plan.walls)
        if abs(w.y_min - half) < eps or abs(w.y_max + half) < eps
    ]


def _glass_walls(plan: floor_plan.FloorPlan, share: float) -> set[int]:
    """The hallway fronts that are glass: round(share * fronts) of them.

    Which ones is seeded by the plan, so the same plan and share always
    give the same glass.
    """
    fronts = hallway_fronts(plan)
    count = round(share * len(fronts))
    rng = np.random.default_rng([plan.seed, 5])
    return {int(i) for i in rng.permutation(fronts)[:count]}


def _add_floor_texture(asset: ET.Element, look: Appearance) -> None:
    """Adds the floor's texture and material for the look's floor kind."""
    first, second = _fmt(*look.floor_rgb), _fmt(*look.floor_rgb2)
    tex = {
        "name": FLOOR_TEXTURE,
        "type": "2d",
        "width": str(TEXTURE_SIZE),
        "height": str(TEXTURE_SIZE),
    }
    if look.floor_kind == "checker":
        tex.update(builtin="checker", rgb1=first, rgb2=second)
    elif look.floor_kind == "tiles":
        # "edge" marks the texture's border, so each repeat is one tile.
        tex.update(builtin="flat", rgb1=first, mark="edge", markrgb=second)
    elif look.floor_kind == "carpet":
        tex.update(
            builtin="flat",
            rgb1=first,
            mark="random",
            markrgb=second,
            random=_fmt(CARPET_FLECKS),
        )
    else:  # plain
        tex.update(builtin="flat", rgb1=first)
    ET.SubElement(asset, "texture", tex)
    # texuniform: texrepeat counts repeats per metre, not per floor, so a
    # tile keeps its size whatever the size of the building.
    repeats = 1.0 / look.floor_period_m
    ET.SubElement(
        asset,
        "material",
        name=FLOOR_MATERIAL,
        texture=FLOOR_TEXTURE,
        texrepeat=_fmt(repeats, repeats),
        texuniform="true",
    )


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
        group=str(ROBOT_GEOM_GROUP),
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
