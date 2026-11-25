"""The data engine: sim frames with exact floor and depth labels (V7).

    pixi run python -m minifsd.data.datagen --split train --plans 100

For each floor plan it samples random collision-free poses, puts the robot
there (no physics), and renders RGB, depth and the floor mask from the front
camera. Labels come from the renderer, so they are exact and free.

Splits are by floor plan: train, val and test draw seeds from disjoint
ranges, so no layout is in two splits and a model can't score well on val by
remembering a floor it trained on.

Each plan gets clutter (furniture-sized boxes, minifsd.world.clutter) as
distractors, and each frame its own camera: from robot height to a handheld
phone tilted down (D1), with its own field of view (D2). Each plan also
gets its own floor pattern and colours (D3), and about half get glass room
fronts (D3b). Lighting varies per plan too (D4), and so does the style
of the room-number signs (D7). Last, each frame goes
through camera effects (D5): blur, noise, exposure, gamma and JPEG.

Output, per split folder:
    plan0000042_0007_rgb.png    RGB
    plan0000042_0007_floor.png  floor mask, 0 or 255
    plan0000042_0007_depth.png  depth, uint16 millimetres (NYUv2/TUM style)
    meta.jsonl                  one line per frame: seed, pose, intrinsics,
                                look, effects, boxes
    coco.json                   the boxes as a COCO detection file (D8)
"""

import argparse
import colorsys
import dataclasses
import json
import math
import pathlib
from collections.abc import Iterator, Sequence

import numpy as np
from scipy import ndimage

from minifsd.data import boxes as boxes_lib
from minifsd.data import camera_effects
from minifsd.geometry import frames
from minifsd.world import clutter as clutter_lib
from minifsd.world import floor_plan, mjcf, occupancy, signs, sim

# Camera positions at least this far from any wall: the robot's radius
# (0.2 m) plus room to spare, so the camera is never inside a wall.
CLEARANCE = 0.35
SPLITS = ("train", "val", "test")
# Floor a pose must see along its heading, in metres beyond where its
# camera's lowest ray meets the floor. Without it, 7 to 10% of frames faced a
# wall and showed almost no floor (D6); a tall camera held level only sees
# floor from 2 to 3 m out. The real photos look along open space, too.
MIN_CLEAR_AHEAD = 1.0
YAW_TRIES = 30  # headings tried at one spot before moving to another
# D1 camera poses: robot height up to a handheld phone (design: real test
# photos at about 0.5 m and 1.5 m), tilted down as a hand holds a phone, and
# a little roll for an unsteady hand.
CAMERA_HEIGHT_RANGE = (0.3, 1.6)
MAX_PITCH = math.radians(25.0)
MAX_ROLL = math.radians(5.0)
# D2 lenses, vertical field of view on our 4:3 image. A phone's main camera
# is about 55 to 65 degrees in 4:3 landscape; the range goes either side so
# models don't learn one focal length.
FOVY_RANGE_DEG = (50.0, 75.0)
# D3 looks. Most plans get muted colours, like real buildings: greys,
# beiges, browns, dull blues. WILD_FRACTION of them get any colour at all,
# which stops a model from learning "floor is a muted colour" (option c).
WILD_FRACTION = 0.2
MUTED_MAX_SATURATION = 0.3  # HSV saturation, 0 grey to 1 pure colour
FLOOR_PERIOD_RANGE_M = (0.3, 2.0)  # small tiles to big checker squares
# HSV value (brightness) per surface. Ceilings are the lightest, walls
# mostly light, floors anything from dark carpet to pale tile.
FLOOR_VALUE_RANGE = (0.15, 0.85)
WALL_VALUE_RANGE = (0.45, 0.95)
CEILING_VALUE_RANGE = (0.75, 1.0)
# D3b: about half the plans have glass room fronts (the campus building has
# many), and those get glass on a share of their fronts.
GLASS_PLAN_SHARE = 0.5
GLASS_SHARE_RANGE = (0.3, 1.0)
# D4 lighting per plan: from a dim corridor to a bright atrium. Ambient has a
# floor above zero so walls seen side-on never go black. The angled light
# comes from any compass direction, 20 to 70 degrees below the horizon.
AMBIENT_RANGE = (0.05, 0.35)
MAIN_LIGHT_RANGE = (0.3, 0.8)
SIDE_LIGHT_RANGE = (0.1, 0.5)
SIDE_LIGHT_ELEVATION_DEG = (20.0, 70.0)
# D7 sign style per building: dark text on a light plate or light on dark,
# half each, so the text always stands out (brightness difference at least
# MIN_SIGN_CONTRAST). Plates follow the same muted or wild colours as walls;
# text stays near neutral, as on real signs.
SIGN_WORD_CHANCE = 0.4
MIN_SIGN_CONTRAST = 0.4
LIGHT_PLATE_VALUE = (0.8, 1.0)
DARK_PLATE_VALUE = (0.1, 0.35)
DARK_TEXT_VALUE = (0.0, 0.2)
LIGHT_TEXT_VALUE = (0.9, 1.0)
# D8 door frames per building: dark wood to pale paint or metal.
DOOR_FRAME_VALUE_RANGE = (0.15, 0.9)
# Each split's seeds start here. Splits can hold up to SPLIT_SIZE plans.
SPLIT_SIZE = 1_000_000
_SPLIT_START = {name: i * SPLIT_SIZE for i, name in enumerate(SPLITS)}


@dataclasses.dataclass(frozen=True, eq=False)
class Frame:
    """One rendered view with its labels."""

    plan_seed: int
    index: int  # position in this plan's pose list
    pose: floor_plan.Pose2D
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, metres along the viewing axis
    floor: np.ndarray  # (H, W) bool
    camera: sim.CameraSetup = sim.CameraSetup()
    appearance: mjcf.Appearance = mjcf.DEFAULT_APPEARANCE
    effects: camera_effects.Effects = camera_effects.Effects()
    boxes: tuple[boxes_lib.Box2D, ...] = ()


def with_effects(
    frame: Frame, effects: camera_effects.Effects, rng: np.random.Generator
) -> Frame:
    """The frame as a real camera would have taken it (D5).

    Args:
        frame: A clean render.
        effects: The camera effects for this frame.
        rng: The random source for sensor noise.

    Returns:
        A new frame with the effects applied to its RGB and recorded. The
        floor mask and depth are the same arrays: they describe the scene,
        which the camera's flaws don't change.
    """
    rgb = camera_effects.apply(frame.rgb, effects, rng)
    return dataclasses.replace(frame, rgb=rgb, effects=effects)


def seeds_for(split: str, n: int) -> list[int]:
    """The floor-plan seeds of a split.

    Args:
        split: "train", "val" or "test".
        n: How many plans, at most SPLIT_SIZE.

    Returns:
        n seeds, none shared with another split.

    Raises:
        ValueError: For an unknown split or too many plans.
    """
    if split not in _SPLIT_START:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if not 0 <= n <= SPLIT_SIZE:
        raise ValueError(f"n must be in [0, {SPLIT_SIZE}], got {n}")
    start = _SPLIT_START[split]
    return list(range(start, start + n))


def sample_poses(
    plan: floor_plan.FloorPlan,
    n: int,
    rng: np.random.Generator,
    clutter: Sequence[clutter_lib.Box] = (),
    cameras: Sequence[sim.CameraSetup] | None = None,
) -> list[floor_plan.Pose2D]:
    """Random poses inside the building, looking along open space.

    Each pose is at least CLEARANCE from walls and clutter. Along its
    heading it has free space out to where its camera sees the floor, plus
    MIN_CLEAR_AHEAD, so the camera never faces a wall with no floor in view.

    Args:
        plan: The floor plan.
        n: How many poses.
        rng: The random source; the same state gives the same poses.
        clutter: Boxes to keep CLEARANCE from, too.
        cameras: The camera each pose will have, one per pose; None for
            the robot's own camera throughout.

    Returns:
        n poses at cell centres of the occupancy grid.

    Raises:
        RuntimeError: If the plan has almost no open view anywhere.
    """
    grid = occupancy.from_floor_plan(plan)
    free = ~grid.occupied
    # Distance from each free cell's centre to the nearest wall cell's
    # centre; the wall's edge is half a cell nearer.
    clear = ndimage.distance_transform_edt(free) * grid.resolution
    clear -= grid.resolution / 2
    # The grid has free border outside the building; keep only the region
    # connected to the spawn point.
    regions, _ = ndimage.label(free)
    row, col = grid.world_to_cell(plan.spawn.x, plan.spawn.y)
    inside = regions == regions[row, col]
    ok = inside & (clear >= CLEARANCE)
    if clutter:
        cols_x = (
            grid.origin_x + (np.arange(ok.shape[1]) + 0.5) * grid.resolution
        )
        rows_y = (
            grid.origin_y + (np.arange(ok.shape[0]) + 0.5) * grid.resolution
        )
        xs, ys = np.meshgrid(cols_x, rows_y)
        for box in clutter:
            ok &= ~clutter_lib.covers(box, xs, ys, CLEARANCE)
    rows, cols = np.nonzero(ok)

    # Cells the camera can see across: free, inside, and off the clutter.
    open_ = free & inside
    if clutter:
        for box in clutter:
            # Grown by half a cell's diagonal: the test is at cell centres,
            # so a box edge between two centres would slip through.
            open_ &= ~clutter_lib.covers(box, xs, ys, grid.resolution * 0.71)
    if cameras is None:
        cameras = [sim.CameraSetup()] * n
    if len(cameras) != n:
        raise ValueError(f"need {n} cameras, got {len(cameras)}")

    def clear_ahead(x: float, y: float, yaw: float, reach: float) -> bool:
        steps = np.arange(0.0, reach, grid.resolution / 2)
        # world_to_cell is for one point; this is it for the whole ray.
        px = x + steps * math.cos(yaw)
        py = y + steps * math.sin(yaw)
        c = np.floor((px - grid.origin_x) / grid.resolution).astype(int)
        r = np.floor((py - grid.origin_y) / grid.resolution).astype(int)
        inside_grid = (
            (r >= 0) & (r < open_.shape[0]) & (c >= 0) & (c < open_.shape[1])
        )
        return bool(inside_grid.all() and open_[r, c].all())

    poses: list[floor_plan.Pose2D] = []
    spots_left = max(n, 1) * YAW_TRIES  # a bound, never reached in practice
    while len(poses) < n:
        spots_left -= 1
        if spots_left < 0:
            raise RuntimeError(f"found only {len(poses)} of {n} open poses")
        cam = cameras[len(poses)]
        reach = _floor_near_edge(cam) + MIN_CLEAR_AHEAD
        i = int(rng.integers(len(rows)))
        x, y = grid.cell_centre(int(rows[i]), int(cols[i]))
        # A few headings at this spot; if none is open, try another spot.
        for _ in range(YAW_TRIES):
            yaw = float(rng.uniform(-math.pi, math.pi))
            if clear_ahead(x, y, yaw, reach):
                poses.append(floor_plan.Pose2D(x, y, yaw))
                break
    return poses


def _floor_near_edge(cam: sim.CameraSetup) -> float:
    """Metres ahead where the bottom edge of the image meets the floor."""
    lowest_ray = cam.pitch + math.radians(cam.fovy_deg) / 2
    return cam.height / math.tan(lowest_ray)


def sample_cameras(n: int, rng: np.random.Generator) -> list[sim.CameraSetup]:
    """Random camera setups within the D1 and D2 ranges, uniform in each.

    Args:
        n: How many.
        rng: The random source; the same state gives the same poses.

    Returns:
        n camera poses.
    """
    heights = rng.uniform(*CAMERA_HEIGHT_RANGE, size=n)
    pitches = rng.uniform(0.0, MAX_PITCH, size=n)
    rolls = rng.uniform(-MAX_ROLL, MAX_ROLL, size=n)
    fovys = rng.uniform(*FOVY_RANGE_DEG, size=n)  # drawn last: D1 unchanged
    return [
        sim.CameraSetup(float(h), float(p), float(r), float(f))
        for h, p, r, f in zip(heights, pitches, rolls, fovys, strict=True)
    ]


def sample_appearances(
    n: int, rng: np.random.Generator
) -> list[mjcf.Appearance]:
    """Random looks: mostly muted colours, WILD_FRACTION in any colour.

    Args:
        n: How many.
        rng: The random source; the same state gives the same looks.

    Returns:
        n looks, each a valid mjcf.Appearance.
    """
    looks = []
    for _ in range(n):
        wild = bool(rng.random() < WILD_FRACTION)
        kind = str(rng.choice(mjcf.FLOOR_KINDS))
        period = float(rng.uniform(*FLOOR_PERIOD_RANGE_M))
        floor_rgb = _colour(rng, wild, FLOOR_VALUE_RANGE)
        floor_rgb2 = _shade(rng, floor_rgb)
        wall_rgb = _colour(rng, wild, WALL_VALUE_RANGE)
        ceiling_rgb = _colour(rng, wild, CEILING_VALUE_RANGE)
        glass = 0.0
        if rng.random() < GLASS_PLAN_SHARE:
            glass = float(rng.uniform(*GLASS_SHARE_RANGE))
        # Lighting is drawn last, so the D3 looks above are unchanged.
        azimuth = rng.uniform(0.0, 2 * math.pi)
        down = math.radians(rng.uniform(*SIDE_LIGHT_ELEVATION_DEG))
        side_dir = (
            float(math.cos(down) * math.cos(azimuth)),
            float(math.cos(down) * math.sin(azimuth)),
            float(-math.sin(down)),
        )
        ambient = float(rng.uniform(*AMBIENT_RANGE))
        main_light = float(rng.uniform(*MAIN_LIGHT_RANGE))
        side_light = float(rng.uniform(*SIDE_LIGHT_RANGE))
        # Drawn last, so the D3 to D4 values above are unchanged.
        sign_rgb, sign_text_rgb, sign_font, sign_word = _sign_style(rng, wild)
        door_frame = _colour(rng, False, DOOR_FRAME_VALUE_RANGE)  # D8, last
        looks.append(
            mjcf.Appearance(
                floor_kind=kind,
                floor_rgb=floor_rgb,
                floor_rgb2=floor_rgb2,
                floor_period_m=period,
                wall_rgb=wall_rgb,
                ceiling_rgb=ceiling_rgb,
                glass_share=glass,
                ambient=ambient,
                main_light=main_light,
                side_light=side_light,
                side_light_dir=side_dir,
                sign_rgb=sign_rgb,
                sign_text_rgb=sign_text_rgb,
                sign_font=sign_font,
                sign_word=sign_word,
                door_frame_rgb=door_frame,
            )
        )
    return looks


def _sign_style(
    rng: np.random.Generator, wild: bool
) -> tuple[mjcf.RGB, mjcf.RGB, str, str]:
    """One building's sign style: plate colour, text colour, font, word."""
    dark_text = rng.random() < 0.5
    plate_value = LIGHT_PLATE_VALUE if dark_text else DARK_PLATE_VALUE
    text_value = DARK_TEXT_VALUE if dark_text else LIGHT_TEXT_VALUE
    text = _colour(rng, False, text_value)
    # HSV value is not brightness as seen: a saturated blue at value 0.8 is
    # still dark. Redraw the plate until the text stands out.
    for _ in range(50):
        plate = _colour(rng, wild, plate_value)
        if abs(_luma(plate) - _luma(text)) >= MIN_SIGN_CONTRAST:
            break
    else:
        # Grey at the far end of the range: white against text of luma at
        # most 0.2, or 0.1 against text of luma at least 0.63 (value 0.9,
        # saturation 0.3). Both clear MIN_SIGN_CONTRAST.
        v = plate_value[1] if dark_text else plate_value[0]
        plate = (v, v, v)
    font = str(rng.choice(list(signs.FONTS)))
    word = (
        str(rng.choice(signs.WORDS)) if rng.random() < SIGN_WORD_CHANCE else ""
    )
    return plate, text, font, word


def _luma(rgb: mjcf.RGB) -> float:
    """Brightness as the eye sees it (Rec. 601 weights)."""
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def _colour(
    rng: np.random.Generator,
    wild: bool,
    value_range: tuple[float, float],
) -> mjcf.RGB:
    """One colour, drawn in HSV so "muted" is just a saturation cap.

    Args:
        rng: The random source.
        wild: True for any hue and saturation; False caps saturation at
            MUTED_MAX_SATURATION.
        value_range: HSV value (brightness) bounds, 0 black to 1 white.

    Returns:
        (r, g, b), each in [0, 1].
    """
    hue = rng.uniform(0.0, 1.0)
    saturation = rng.uniform(0.0, 1.0 if wild else MUTED_MAX_SATURATION)
    value = rng.uniform(*value_range)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (float(r), float(g), float(b))


def _shade(rng: np.random.Generator, rgb: mjcf.RGB) -> mjcf.RGB:
    """A darker or lighter shade of the same hue, for the second floor colour.

    Grout, carpet flecks and the other checker square are usually the floor's
    own colour, darker or lighter, not an unrelated one.

    Args:
        rng: The random source.
        rgb: The main floor colour.

    Returns:
        (r, g, b), each in [0, 1], different from rgb.
    """
    if rng.random() < 0.5:
        k = rng.uniform(0.4, 0.75)  # darker: scale towards black
        return (float(rgb[0] * k), float(rgb[1] * k), float(rgb[2] * k))
    t = rng.uniform(0.3, 0.6)  # lighter: mix towards white
    return (
        float(rgb[0] + (1 - rgb[0]) * t),
        float(rgb[1] + (1 - rgb[1]) * t),
        float(rgb[2] + (1 - rgb[2]) * t),
    )


def render_frames(
    plan: floor_plan.FloorPlan,
    poses: Sequence[floor_plan.Pose2D],
    clutter: Sequence[clutter_lib.Box] = (),
    cameras: Sequence[sim.CameraSetup] | None = None,
    appearance: mjcf.Appearance = mjcf.DEFAULT_APPEARANCE,
) -> Iterator[Frame]:
    """Renders one frame per pose.

    Args:
        plan: The floor plan.
        poses: Where to put the robot.
        clutter: Boxes standing in the world (distractors).
        cameras: One camera pose per robot pose; None for the robot's own
            camera throughout.
        appearance: The plan's look, the same for all its frames.

    Yields:
        The frames, in pose order.
    """
    if cameras is None:
        cameras = [sim.CameraSetup()] * len(poses)
    s = sim.Sim(plan, clutter=clutter, appearance=appearance)
    labelled = s.labelled_objects()
    objects = {g: v for g, v in labelled.items() if v[0] != "door"}
    doors = {g: v for g, v in labelled.items() if v[0] == "door"}
    try:
        for i, (pose, cam) in enumerate(zip(poses, cameras, strict=True)):
            s.set_pose(pose)
            s.set_camera(cam)
            yield Frame(
                plan_seed=plan.seed,
                index=i,
                pose=pose,
                rgb=s.render_rgb(),
                depth=s.render_depth(),
                floor=s.render_floor_mask(),
                camera=cam,
                appearance=appearance,
                boxes=_boxes(s, objects, doors),
            )
    finally:
        s.close()


def _boxes(
    s: sim.Sim,
    objects: dict[int, tuple[str, str]],
    doors: dict[int, tuple[str, str]],
) -> tuple[boxes_lib.Box2D, ...]:
    """The frame's boxes, from two label renders (see sim.*_LABEL_GROUPS)."""
    found = boxes_lib.boxes_from_ids(
        s.render_segmentation(groups=sim.OBJECT_LABEL_GROUPS), objects
    )
    found += boxes_lib.boxes_from_ids(
        s.render_segmentation(groups=sim.DOOR_LABEL_GROUPS), doors
    )
    return tuple(found)


def write_frame(frame: Frame, out_dir: pathlib.Path) -> None:
    """Writes the frame's images and appends its line to meta.jsonl.

    Args:
        frame: The frame.
        out_dir: The split's folder; created if missing.
    """
    import cv2  # only the writer needs OpenCV

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"plan{frame.plan_seed:07d}_{frame.index:04d}"
    cv2.imwrite(
        str(out_dir / f"{stem}_rgb.png"),
        cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(out_dir / f"{stem}_floor.png"), frame.floor.astype(np.uint8) * 255
    )
    # Millimetres in 16 bits: up to 65.5 m at 1 mm steps, lossless in PNG.
    depth_mm = np.clip(np.round(frame.depth * 1000), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out_dir / f"{stem}_depth.png"), depth_mm)

    k = frames.intrinsics_from_fovy(
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, frame.camera.fovy_deg
    )
    meta = {
        "stem": stem,
        "plan_seed": frame.plan_seed,
        "index": frame.index,
        "pose": [frame.pose.x, frame.pose.y, frame.pose.yaw],
        "camera_height": frame.camera.height,
        "camera_pitch": frame.camera.pitch,
        "camera_roll": frame.camera.roll,
        "intrinsics": dataclasses.asdict(k),
        "appearance": dataclasses.asdict(frame.appearance),
        "effects": dataclasses.asdict(frame.effects),
        "boxes": [dataclasses.asdict(b) for b in frame.boxes],
    }
    with (out_dir / "meta.jsonl").open("a") as f:
        f.write(json.dumps(meta) + "\n")


def write_coco(out_dir: pathlib.Path) -> None:
    """Writes the split's boxes, from meta.jsonl, as out_dir/coco.json.

    Args:
        out_dir: A split folder written by write_frame.
    """
    images = []
    with (out_dir / "meta.jsonl").open() as f:
        for line in f:
            meta = json.loads(line)
            found = [boxes_lib.Box2D(**b) for b in meta["boxes"]]
            images.append(
                (
                    f"{meta['stem']}_rgb.png",
                    mjcf.IMAGE_WIDTH,
                    mjcf.IMAGE_HEIGHT,
                    found,
                )
            )
    with (out_dir / "coco.json").open("w") as f:
        json.dump(boxes_lib.to_coco(images), f)


@dataclasses.dataclass(frozen=True, eq=False)
class StoredFrame:
    """A frame read back from disk, shaped like nyuv2.Sample for the evals."""

    index: int  # line number in meta.jsonl
    stem: str
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 metres
    floor: np.ndarray  # (H, W) bool
    labeled: np.ndarray  # (H, W) bool, all True: the renderer labels everything


def read_frames(folder: pathlib.Path) -> Iterator[StoredFrame]:
    """Reads a split written by write_frame, one frame at a time.

    Args:
        folder: The split's folder, holding meta.jsonl.

    Yields:
        The frames in the order they were written.
    """
    import cv2

    def read(path: pathlib.Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(path)
        return image

    with (folder / "meta.jsonl").open() as lines:
        for i, line in enumerate(lines):
            stem = json.loads(line)["stem"]
            rgb = cv2.cvtColor(
                read(folder / f"{stem}_rgb.png"), cv2.COLOR_BGR2RGB
            )
            floor = read(folder / f"{stem}_floor.png") > 0
            depth = (read(folder / f"{stem}_depth.png") / 1000.0).astype(
                np.float32
            )
            yield StoredFrame(
                index=i,
                stem=stem,
                rgb=rgb,
                depth=depth,
                floor=floor,
                labeled=np.ones(floor.shape, dtype=bool),
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Renders a split: N floor plans, K frames each."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--plans", type=int, default=10)
    parser.add_argument("--frames-per-plan", type=int, default=20)
    parser.add_argument(
        "--clutter", type=int, default=6, help="boxes per floor plan"
    )
    parser.add_argument("--out", default="data/sim/v7", type=pathlib.Path)
    args = parser.parse_args(argv)

    out = args.out / args.split
    if (out / "meta.jsonl").exists():
        raise SystemExit(f"{out} already has frames; pick a new --out")
    total = 0
    for seed in seeds_for(args.split, args.plans):
        # Layout size varies with the seed too, from 2 to 6 rooms a side.
        rooms = int(np.random.default_rng([seed, 0]).integers(2, 7))
        plan = floor_plan.generate_floor_plan(seed, rooms_per_side=rooms)
        boxes = clutter_lib.generate_clutter(
            plan, args.clutter, np.random.default_rng([seed, 2])
        )
        # Cameras first: how much open space a pose needs depends on them.
        cameras = sample_cameras(
            args.frames_per_plan, np.random.default_rng([seed, 3])
        )
        rng = np.random.default_rng([seed, 1])
        poses = sample_poses(plan, args.frames_per_plan, rng, boxes, cameras)
        (look,) = sample_appearances(1, np.random.default_rng([seed, 4]))
        fx_rng = np.random.default_rng([seed, 6])
        for frame in render_frames(plan, poses, boxes, cameras, look):
            effects = camera_effects.sample_effects(fx_rng)
            write_frame(with_effects(frame, effects, fx_rng), out)
            total += 1
    write_coco(out)
    print(f"{total} frames from {args.plans} plans in {out}/")


if __name__ == "__main__":
    main()
