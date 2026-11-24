"""The data engine: sim frames with exact floor and depth labels (V7).

    pixi run python -m minifsd.data.datagen --split train --plans 100

For each floor plan it samples random collision-free poses, puts the robot
there (no physics), and renders RGB, depth and the floor mask from the front
camera. Labels come from the renderer, so they are exact and free.

Splits are by floor plan: train, val and test draw seeds from disjoint
ranges, so no layout is in two splits and a model can't score well on val by
remembering a floor it trained on.

Each plan gets clutter (furniture-sized boxes, minifsd.world.clutter) as
distractors. Camera height, textures and lighting are still fixed; the rest
of domain randomization is M2.

Output, per split folder:
    plan0000042_0007_rgb.png    RGB
    plan0000042_0007_floor.png  floor mask, 0 or 255
    plan0000042_0007_depth.png  depth, uint16 millimetres (NYUv2/TUM style)
    meta.jsonl                  one line per frame: seed, pose, intrinsics
"""

import argparse
import dataclasses
import json
import math
import pathlib
from collections.abc import Iterator, Sequence

import numpy as np
from scipy import ndimage

from minifsd.geometry import frames
from minifsd.world import clutter as clutter_lib
from minifsd.world import floor_plan, mjcf, occupancy, sim

# Camera positions at least this far from any wall: the robot's radius
# (0.2 m) plus room to spare, so the camera is never inside a wall.
CLEARANCE = 0.35
SPLITS = ("train", "val", "test")
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
) -> list[floor_plan.Pose2D]:
    """Random poses inside the building, at least CLEARANCE from walls.

    Args:
        plan: The floor plan.
        n: How many poses.
        rng: The random source; the same state gives the same poses.
        clutter: Boxes to keep CLEARANCE from, too.

    Returns:
        n poses at cell centres of the occupancy grid, with any yaw.
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
    picks = rng.choice(len(rows), size=n, replace=len(rows) < n)
    poses = []
    for i in picks:
        x, y = grid.cell_centre(int(rows[i]), int(cols[i]))
        yaw = float(rng.uniform(-math.pi, math.pi))
        poses.append(floor_plan.Pose2D(x, y, yaw))
    return poses


def render_frames(
    plan: floor_plan.FloorPlan,
    poses: Sequence[floor_plan.Pose2D],
    clutter: Sequence[clutter_lib.Box] = (),
) -> Iterator[Frame]:
    """Renders one frame per pose.

    Args:
        plan: The floor plan.
        poses: Where to put the robot.
        clutter: Boxes standing in the world (distractors).

    Yields:
        The frames, in pose order.
    """
    s = sim.Sim(plan, clutter=clutter)
    try:
        for i, pose in enumerate(poses):
            s.set_pose(pose)
            yield Frame(
                plan_seed=plan.seed,
                index=i,
                pose=pose,
                rgb=s.render_rgb(),
                depth=s.render_depth(),
                floor=s.render_floor_mask(),
            )
    finally:
        s.close()


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
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, mjcf.FOVY_DEG
    )
    meta = {
        "stem": stem,
        "plan_seed": frame.plan_seed,
        "index": frame.index,
        "pose": [frame.pose.x, frame.pose.y, frame.pose.yaw],
        "camera_height": mjcf.CAMERA_HEIGHT,
        "intrinsics": dataclasses.asdict(k),
    }
    with (out_dir / "meta.jsonl").open("a") as f:
        f.write(json.dumps(meta) + "\n")


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
        rng = np.random.default_rng([seed, 1])
        poses = sample_poses(plan, args.frames_per_plan, rng, boxes)
        for frame in render_frames(plan, poses, boxes):
            write_frame(frame, out)
            total += 1
    print(f"{total} frames from {args.plans} plans in {out}/")


if __name__ == "__main__":
    main()
