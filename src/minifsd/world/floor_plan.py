"""Seeded floor plans: one straight hallway with numbered rooms on both sides.

A floor plan is plain data: wall rectangles, rooms with doors, and a spawn pose.
Later steps turn it into a MuJoCo model (R2), an occupancy grid for Nav2 (R7)
and mission goals (R9). The same seed always gives the same floor plan, which is
what lets the dataset be split by seed.

Coordinates are metres in the world frame. The hallway runs along +x from
x = 0 to x = length and is centred on y = 0. Rooms on the left (+y) get odd
numbers, rooms on the right (-y) get even numbers.

      y
      ^   +-----+-----+-----+
      |   | 101 | 103 | 105 |
      |   +-- --+-- --+-- --+   <- door gaps in the hallway wall
      |   S->   hallway     |
      |   +-- --+-- --+-- --+
      |   | 102 | 104 | 106 |
      |   +-----+-----+-----+
      +-------------------------> x
"""

import dataclasses
import math

import numpy as np

WALL_THICKNESS = 0.1
GOAL_STANDOFF = 0.6  # distance from the door to the goal pose, in metres
MAX_ROOMS_PER_SIDE = 49  # keeps room numbers inside the floor's hundred

# Ranges for the random choices, in metres. Picked to look like ordinary
# university buildings; the generator samples uniformly inside each one.
HALLWAY_WIDTH_RANGE = (1.8, 3.0)
ROOM_WIDTH_RANGE = (3.0, 4.5)
ROOM_DEPTH_RANGE = (3.0, 5.0)
DOOR_WIDTH_RANGE = (0.9, 1.2)

LEFT = 1
RIGHT = -1


@dataclasses.dataclass(frozen=True)
class Pose2D:
    """A position and heading on the floor.

    Attributes:
        x: Metres.
        y: Metres.
        yaw: Radians, counterclockwise from +x.
    """

    x: float
    y: float
    yaw: float


@dataclasses.dataclass(frozen=True)
class Wall:
    """An axis-aligned wall footprint. Height is set later, in the MJCF."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclasses.dataclass(frozen=True)
class Room:
    """A room and the door that leads into it from the hallway.

    Attributes:
        number: Room number, for example 103.
        door_x: x of the door centre, on the hallway wall.
        side: LEFT (+y) or RIGHT (-y) of the hallway.
        goal: Where the robot stops: in the hallway, facing the door.
    """

    number: int
    door_x: float
    side: int
    goal: Pose2D


@dataclasses.dataclass(frozen=True)
class FloorPlan:
    """A generated floor plan. Frozen, so two floor plans compare with ==."""

    seed: int
    length: float
    hallway_width: float
    door_width: float
    walls: tuple[Wall, ...]
    rooms: tuple[Room, ...]
    spawn: Pose2D


def split_wall(
    x_start: float, x_end: float, gaps: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Cuts door gaps out of a straight wall.

    Example: a wall from 0 to 10 with one gap at (2, 3) leaves the solid
    pieces [(0, 2), (3, 10)].

    Args:
        x_start: Where the wall starts.
        x_end: Where the wall ends, greater than x_start.
        gaps: Door gaps as (a, b) with x_start <= a < b <= x_end. Gaps never
            overlap each other but may come in any order.

    Returns:
        The solid pieces as (a, b) pairs, sorted by a. Pieces of zero length
        (a gap touching the wall's end, for example) are left out.
    """
    pieces = []
    pos = x_start  # where the next solid piece starts
    # Sorted, so the walk goes left to right whatever order the gaps came in.
    for a, b in sorted(gaps):
        if a > pos:  # a gap that starts exactly at pos leaves no piece
            pieces.append((pos, a))
        pos = b
    if x_end > pos:
        pieces.append((pos, x_end))
    return pieces


def room_number(floor_number: int, index: int, side: int) -> int:
    """Returns the number of the index-th room along +x on one side.

    Left rooms are odd and right rooms are even, like most real buildings:
    on floor 1, index 0 gives 101 on the left and 102 on the right, index 1
    gives 103 and 104.

    Args:
        floor_number: 1 or more. It becomes the hundreds digit, so floor 14
            gives 1401, 1402 and so on.
        index: 0 for the room nearest x = 0, counting up along +x.
        side: LEFT or RIGHT.
    """
    # Each index uses up one odd and one even number: index i gets 2i + 1 on
    # the left and 2i + 2 on the right.
    offset = 1 if side == LEFT else 2
    return floor_number * 100 + 2 * index + offset


def generate_floor_plan(
    seed: int, rooms_per_side: int = 4, floor_number: int = 1
) -> FloorPlan:
    """Generates a hallway with numbered rooms on both sides.

    Args:
        seed: Decides every random choice. Same seed, same floor_plan.
        rooms_per_side: Rooms on each side of the hallway, 1 to 49.
        floor_number: Sets the hundreds digit of each room number.

    Returns:
        The floor_plan. Its hallway walls have one door gap per room; every
        other wall is solid, so the whole floor is closed.

    Raises:
        ValueError: If rooms_per_side or floor_number is out of range.
    """
    if not 1 <= rooms_per_side <= MAX_ROOMS_PER_SIDE:
        raise ValueError(
            f"rooms_per_side must be 1 to {MAX_ROOMS_PER_SIDE}, "
            f"got {rooms_per_side}"
        )
    if floor_number < 1:
        raise ValueError(f"floor_number must be 1 or more, got {floor_number}")

    # One generator, created from the seed, makes every random choice. The
    # global `random` module would let unrelated code change the result.
    rng = np.random.default_rng(seed)
    hallway_width = float(rng.uniform(*HALLWAY_WIDTH_RANGE))
    room_width = float(rng.uniform(*ROOM_WIDTH_RANGE))
    room_depth = float(rng.uniform(*ROOM_DEPTH_RANGE))
    door_width = float(rng.uniform(*DOOR_WIDTH_RANGE))

    t = WALL_THICKNESS
    half = hallway_width / 2  # y of each hallway wall's inner face
    length = rooms_per_side * room_width
    outer = half + t + room_depth + t  # |y| of the back walls' outer face

    door_xs = [(i + 0.5) * room_width for i in range(rooms_per_side)]
    gaps = [(x - door_width / 2, x + door_width / 2) for x in door_xs]

    walls: list[Wall] = []
    for side in (LEFT, RIGHT):
        # Hallway wall on this side, with one door gap per room.
        for a, b in split_wall(0.0, length, gaps):
            walls.append(_wall_band(a, b, side, half, half + t))
        # Back wall of this side's rooms.
        walls.append(_wall_band(0.0, length, side, outer - t, outer))
        # Walls between neighbouring rooms.
        for i in range(1, rooms_per_side):
            x = i * room_width
            walls.append(
                _wall_band(x - t / 2, x + t / 2, side, half + t, outer - t)
            )
    # End walls close the hallway and the end rooms at x = 0 and x = length.
    walls.append(Wall(-t, -outer, 0.0, outer))
    walls.append(Wall(length, -outer, length + t, outer))

    rooms = []
    for i, x in enumerate(door_xs):
        for side in (LEFT, RIGHT):
            goal = Pose2D(
                x=x,
                y=side * (half - GOAL_STANDOFF),
                yaw=side * math.pi / 2,  # facing the door
            )
            rooms.append(
                Room(room_number(floor_number, i, side), x, side, goal)
            )

    return FloorPlan(
        seed=seed,
        length=length,
        hallway_width=hallway_width,
        door_width=door_width,
        walls=tuple(walls),
        rooms=tuple(rooms),
        spawn=Pose2D(x=1.0, y=0.0, yaw=0.0),
    )


def _wall_band(
    x_min: float, x_max: float, side: int, near: float, far: float
) -> Wall:
    """A wall spanning near <= |y| <= far on one side of the hallway."""
    if side == LEFT:
        return Wall(x_min, near, x_max, far)
    return Wall(x_min, -far, x_max, -near)
