"""Clutter: furniture-sized boxes that are in the world but not on the map.

Nav2 plans on the floor plan's walls (R7). Clutter is left off that map on
purpose, so the robot can only learn about it through the camera (R10, M4):
that is the "dodge" test. The data engine also uses it as distractors (M2).

Placement keeps the floor usable: boxes don't overlap walls, stay clear of
the spawn point, and never cut a room off. After each box, a grid search
checks that every room's goal can still be reached from spawn by a robot
needing CLEARANCE metres on each side.

Kinds (D11): by default every piece is a solid box, as before. The data
engine also asks for tables (a top on four legs) and desks (a top on two
side panels), whose tops are flat surfaces above the floor with floor
visible underneath, the case SegFormer got wrong on real rooms (E2a). Boxes
stand in for counters, beds and sofas. A piece's footprint is always an
obstacle, so the robot never plans under a table.
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from minifsd.world import floor_plan, mjcf, occupancy

MIN_HALF = 0.15  # half-lengths: boxes are 0.3 to 1.2 m across
MAX_HALF = 0.6
MIN_HEIGHT = 0.3  # a low stool up to a counter
MAX_HEIGHT = 1.0
SPAWN_CLEARANCE = 1.0  # free radius around the spawn point
# Each side of the robot's path: its radius plus margin for the controller.
# Gaps narrower than 2 x 0.35 m count as blocked.
CLEARANCE = mjcf.ROBOT_RADIUS + 0.15
TRIES_PER_BOX = 40

FURNITURE_KINDS = ("box", "table", "desk")
# Height and half-length ranges per kind, in metres.
HEIGHTS = {
    "box": (MIN_HEIGHT, MAX_HEIGHT),
    "table": (0.7, 1.05),  # dining and meeting tables up to bar height
    "desk": (0.72, 0.78),
}
HALVES = {
    "box": (MIN_HALF, MAX_HALF),
    "table": (0.3, 0.9),
    "desk": (0.3, 0.8),
}

# Muted furniture colours: wood, grey, dark blue, beige, black, green.
PALETTE = (
    (0.55, 0.40, 0.25, 1.0),
    (0.45, 0.45, 0.47, 1.0),
    (0.20, 0.25, 0.40, 1.0),
    (0.80, 0.75, 0.62, 1.0),
    (0.12, 0.12, 0.12, 1.0),
    (0.30, 0.42, 0.30, 1.0),
)


@dataclasses.dataclass(frozen=True)
class Box:
    """An upright box standing on the floor, turned by yaw about z."""

    x: float
    y: float
    yaw: float
    half_x: float  # half-lengths along the box's own x and y axes
    half_y: float
    height: float
    rgba: tuple[float, float, float, float]  # body, or legs and sides
    kind: str = "box"  # one of FURNITURE_KINDS
    surface: int = -1  # index of the look's furniture surface; -1: rgba


def _local(
    box: Box, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # World point to the box's own frame: subtract the centre, rotate by -yaw.
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    dx, dy = x - box.x, y - box.y
    return c * dx + s * dy, -s * dx + c * dy


def covers(
    box: Box, x: np.ndarray, y: np.ndarray, grow: float = 0.0
) -> np.ndarray:
    """Element-wise contains() over arrays of points (e.g. grid cells)."""
    lx, ly = _local(box, x, y)
    return (np.abs(lx) <= box.half_x + grow) & (np.abs(ly) <= box.half_y + grow)


def contains(box: Box, x: float, y: float, grow: float = 0.0) -> bool:
    """Whether a point is inside the box's footprint, grown by grow metres.

    Growing each side by grow is slightly generous at the corners (a square
    corner instead of a round one), which errs on the safe side.
    """
    return bool(covers(box, np.asarray(x), np.asarray(y), grow))


@dataclasses.dataclass(frozen=True, eq=False)
class _Floor:
    """The grid and the cells a robot centre may use, walls only."""

    grid: occupancy.Grid
    xs: np.ndarray  # cell-centre coordinates, same shape as the grid
    ys: np.ndarray
    free: np.ndarray  # far enough from walls and inside the building
    inside: np.ndarray  # inside the building at all

    @classmethod
    def of(cls, plan: floor_plan.FloorPlan) -> "_Floor":
        grid = occupancy.from_floor_plan(plan)
        rows, cols = grid.occupied.shape
        res = grid.resolution
        xs = grid.origin_x + (np.arange(cols) + 0.5) * res
        ys = grid.origin_y + (np.arange(rows) + 0.5) * res
        xs, ys = np.meshgrid(xs, ys)
        open_ = ~grid.occupied
        regions, _ = ndimage.label(open_)
        inside = (
            regions == regions[grid.world_to_cell(plan.spawn.x, plan.spawn.y)]
        )
        # Distance from a cell centre to the nearest wall edge.
        clear = ndimage.distance_transform_edt(open_) * res - res / 2
        return cls(grid, xs, ys, inside & (clear >= CLEARANCE), inside)


def _reachable(
    plan: floor_plan.FloorPlan, floor: _Floor, boxes: Sequence[Box]
) -> bool:
    free = floor.free.copy()
    for box in boxes:
        free &= ~covers(box, floor.xs, floor.ys, CLEARANCE)
    regions, _ = ndimage.label(free)
    start = regions[floor.grid.world_to_cell(plan.spawn.x, plan.spawn.y)]
    if start == 0:
        return False  # the spawn itself is blocked
    goals = [floor.grid.world_to_cell(r.goal.x, r.goal.y) for r in plan.rooms]
    return all(regions[g] == start for g in goals)


def reachable(plan: floor_plan.FloorPlan, boxes: Sequence[Box]) -> bool:
    """Whether every room goal can be reached from spawn past the boxes.

    Args:
        plan: The floor plan.
        boxes: Clutter on it.

    Returns:
        True if, on the occupancy grid, a robot needing CLEARANCE on each
        side can get from the spawn to every room's goal.
    """
    return _reachable(plan, _Floor.of(plan), boxes)


def generate_clutter(
    plan: floor_plan.FloorPlan,
    n: int,
    rng: np.random.Generator,
    kinds: Sequence[str] = ("box",),
    surfaces: int = 0,
) -> tuple[Box, ...]:
    """Places up to n boxes, never cutting a room off.

    Args:
        plan: The floor plan.
        n: How many boxes to try for.
        rng: The random source; the same state gives the same boxes.
        kinds: Kinds to draw from, each from FURNITURE_KINDS. With the
            default plain boxes and no surfaces, nothing extra is drawn,
            so the boxes are the same as before D11.
        surfaces: How many furniture surfaces the look has
            (mjcf.Appearance.surfaces); each piece takes one at random.
            0 keeps the plain rgba.

    Returns:
        The boxes placed. Fewer than n only if a floor is too cramped to
        fit them within TRIES_PER_BOX tries each.
    """
    unknown = set(kinds) - set(FURNITURE_KINDS)
    if unknown or not kinds:
        raise ValueError(f"kinds must come from {FURNITURE_KINDS}: {kinds}")
    plain = tuple(kinds) == ("box",) and surfaces == 0
    floor = _Floor.of(plan)
    walls = floor.grid.occupied
    rows, cols = np.nonzero(floor.inside)
    boxes: list[Box] = []
    for _ in range(n * TRIES_PER_BOX):
        if len(boxes) == n:
            break
        i = rng.integers(len(rows))
        box = Box(
            x=float(floor.xs[rows[i], cols[i]]),
            y=float(floor.ys[rows[i], cols[i]]),
            yaw=float(rng.uniform(-math.pi, math.pi)),
            half_x=float(rng.uniform(MIN_HALF, MAX_HALF)),
            half_y=float(rng.uniform(MIN_HALF, MAX_HALF)),
            height=float(rng.uniform(MIN_HEIGHT, MAX_HEIGHT)),
            rgba=PALETTE[int(rng.integers(len(PALETTE)))],
        )
        if not plain:
            # Drawn after the old values, so plain boxes are unchanged.
            kind = str(rng.choice(list(kinds)))
            low, high = HALVES[kind]
            box = dataclasses.replace(
                box,
                half_x=float(rng.uniform(low, high)),
                half_y=float(rng.uniform(low, high)),
                height=float(rng.uniform(*HEIGHTS[kind])),
                kind=kind,
                surface=int(rng.integers(surfaces)) if surfaces else -1,
            )
        footprint = covers(box, floor.xs, floor.ys, 0.0)
        if (footprint & walls).any() or (footprint & ~floor.inside).any():
            continue
        if contains(box, plan.spawn.x, plan.spawn.y, SPAWN_CLEARANCE):
            continue
        if _reachable(plan, floor, [*boxes, box]):
            boxes.append(box)
    return tuple(boxes)
