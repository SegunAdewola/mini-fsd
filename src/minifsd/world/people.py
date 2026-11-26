"""Person-sized walkers that pace the hallway (M4.7).

The sim has no people to render (rendered humans look little like real
ones, so the detector learned people from COCO), but the navigation loop
still has to deal with things that move. A walker is a capsule the size of
a person that paces back and forth along a stretch of hallway.

Walkers yield: one stands still while its next step would bring it within
KEEP_AWAY of the robot, and walks on once the way is clear. They don't
steer around the robot. So a walker never walks into the robot, and any
contact in M4.8 is the robot driving into someone, which is what the
experiment measures.

Walker 0 keeps to the left half of the hallway and walker 1 to the right,
so two walkers never meet, and each lane leaves the other half open.
"""

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

from minifsd.world import clutter as clutter_lib
from minifsd.world import floor_plan, mjcf

RADIUS = mjcf.PERSON_RADIUS
HEIGHT = mjcf.PERSON_HEIGHT
MARGIN = 0.1  # m: gap kept between a walker and the hallway walls
SPAWN_GAP = 2.0  # m: walkers start this far past the robot's spawn, in x
SPEED_RANGE = (0.5, 1.2)  # m/s: a slow walk to a normal one
PATH_RANGE = (3.0, 8.0)  # m: length of the stretch each walker paces
TRIES = 40
# Centre-to-centre distance a walker won't close to: the two radii plus a
# hand's width.
KEEP_AWAY = RADIUS + mjcf.ROBOT_RADIUS + 0.1


@dataclasses.dataclass(frozen=True)
class Walker:
    """One walker's path.

    Attributes:
        x_start: Near end of its stretch of hallway, x in metres.
        x_end: Far end, greater than x_start.
        y: Its lane, metres from the hallway's centre line.
        speed: Walking speed, m/s.
        phase: Metres walked along the out-and-back loop at time 0, from 0
            to twice the stretch's length.
    """

    x_start: float
    x_end: float
    y: float
    speed: float
    phase: float

    def x_at(self, walked: float) -> float:
        """Its x after walking `walked` metres around the out-and-back loop."""
        length = self.x_end - self.x_start
        s = walked % (2 * length)
        return self.x_start + (s if s <= length else 2 * length - s)


class Crowd:
    """Walkers and how far each has walked."""

    def __init__(self, walkers: Sequence[Walker]) -> None:
        """Puts each walker at its phase."""
        self.walkers = tuple(walkers)
        self._walked = [w.phase for w in self.walkers]

    def positions(self) -> list[tuple[float, float]]:
        """Each walker's (x, y) on the floor."""
        return [
            (w.x_at(s), w.y)
            for w, s in zip(self.walkers, self._walked, strict=True)
        ]

    def advance(self, dt: float, robot: tuple[float, float]) -> None:
        """Walks every walker on by dt seconds, except those that must yield.

        Args:
            dt: Seconds.
            robot: The robot's (x, y).
        """
        for i, w in enumerate(self.walkers):
            walked = self._walked[i] + w.speed * dt
            now = math.hypot(w.x_at(self._walked[i]) - robot[0], w.y - robot[1])
            then = math.hypot(w.x_at(walked) - robot[0], w.y - robot[1])
            if then < KEEP_AWAY and then < now:
                continue  # stand still until the robot is out of the way
            self._walked[i] = walked


def generate_walkers(
    plan: floor_plan.FloorPlan,
    rng: np.random.Generator,
    count: int,
    clutter: Sequence[clutter_lib.Box] = (),
) -> tuple[Walker, ...]:
    """Places up to `count` walkers (at most 2, one per side of the centre).

    Args:
        plan: The floor plan.
        rng: The random source; the same state gives the same walkers.
        count: How many to try for, 0 to 2.
        clutter: Boxes no walker's path may cross.

    Returns:
        The walkers that fit. A short hallway, or one crowded with boxes,
        may fit fewer than asked.

    Raises:
        ValueError: If count is not 0, 1 or 2.
    """
    if not 0 <= count <= 2:
        raise ValueError(f"count must be 0 to 2, got {count}")
    half = plan.hallway_width / 2
    lane_far = half - MARGIN - RADIUS  # outermost lane that clears the wall
    x_min = plan.spawn.x + SPAWN_GAP
    x_max = plan.length - MARGIN - RADIUS
    walkers = []
    for i in range(count):
        side = floor_plan.LEFT if i == 0 else floor_plan.RIGHT
        for _ in range(TRIES):
            y = side * float(rng.uniform(min(half / 2, lane_far), lane_far))
            length = float(rng.uniform(*PATH_RANGE))
            length = min(length, x_max - x_min)
            if length < PATH_RANGE[0] / 2:
                break  # the hallway is too short for anyone to pace
            x_start = float(rng.uniform(x_min, x_max - length))
            walker = Walker(
                x_start=x_start,
                x_end=x_start + length,
                y=y,
                speed=float(rng.uniform(*SPEED_RANGE)),
                phase=float(rng.uniform(0.0, 2 * length)),
            )
            if _clear(walker, clutter):
                walkers.append(walker)
                break
    return tuple(walkers)


def _clear(walker: Walker, clutter: Sequence[clutter_lib.Box]) -> bool:
    """True if no box is within RADIUS of the walker's path."""
    xs = np.arange(walker.x_start, walker.x_end + 0.05, 0.05)
    return not any(
        clutter_lib.contains(box, float(x), walker.y, grow=RADIUS)
        for box in clutter
        for x in xs
    )
