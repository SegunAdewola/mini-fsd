"""The closed-loop experiment's records and table (M4.8).

The same missions run with the sim's exact depth (truth) and with the
camera models (model). Each mission records how it ended, how long it took
and how many times the robot bumped into something. A bump is the start of
a run of contact, so one long scrape along a wall is one collision, not one
per tick (Bumps).

The ROS side (tools/closed_loop.py) runs the missions and appends one
Mission per line to a JSON lines file; summarize turns them into the
report's table.
"""

import dataclasses
import json
import pathlib
import statistics
from collections.abc import Sequence

import numpy as np

from minifsd.world import floor_plan

OUTCOMES = ("succeeded", "failed", "timeout")


class Bumps:
    """Counts collisions from a stream of per-tick contact counts."""

    def __init__(self) -> None:
        """Starts at zero, not touching anything."""
        self.total = 0
        self._touching = False

    def update(self, contacts: int) -> None:
        """Takes this tick's contact count; a new touch adds one."""
        touching = contacts > 0
        if touching and not self._touching:
            self.total += 1
        self._touching = touching


@dataclasses.dataclass(frozen=True)
class Mission:
    """One mission's result.

    Attributes:
        condition: "truth" or "model", the perception source.
        seed: Floor plan seed.
        room: Room number asked for.
        outcome: One of OUTCOMES.
        seconds: Sim time from sending the mission to its end.
        collisions: Bumps during the mission.
    """

    condition: str
    seed: int
    room: int
    outcome: str
    seconds: float
    collisions: int


@dataclasses.dataclass(frozen=True)
class Row:
    """One condition's line in the table.

    Attributes:
        condition: The perception source.
        missions: How many ran.
        success_rate: Share that reached the goal.
        clean_success_rate: Share that reached it without a collision.
        median_seconds: Median time of the successes; None without any.
        collisions_per_mission: Mean bumps per mission.
    """

    condition: str
    missions: int
    success_rate: float
    clean_success_rate: float
    median_seconds: float | None
    collisions_per_mission: float


def summarize(missions: Sequence[Mission]) -> list[Row]:
    """One row per condition, in the order the conditions first appear."""
    conditions = list(dict.fromkeys(m.condition for m in missions))
    rows = []
    for condition in conditions:
        mine = [m for m in missions if m.condition == condition]
        wins = [m for m in mine if m.outcome == "succeeded"]
        n = len(mine)
        rows.append(
            Row(
                condition=condition,
                missions=n,
                success_rate=len(wins) / n,
                clean_success_rate=sum(m.collisions == 0 for m in wins) / n,
                median_seconds=(
                    statistics.median(m.seconds for m in wins) if wins else None
                ),
                collisions_per_mission=sum(m.collisions for m in mine) / n,
            )
        )
    return rows


def append(mission: Mission, path: pathlib.Path) -> None:
    """Adds one mission to a JSON lines file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(dataclasses.asdict(mission)) + "\n")


def load(path: pathlib.Path) -> list[Mission]:
    """Reads every mission from a JSON lines file."""
    return [
        Mission(**json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def pick_rooms(plan: floor_plan.FloorPlan, n: int, seed: int) -> list[int]:
    """N distinct room numbers from the plan, the same for the same seed.

    Both conditions run the same list, so they face the same missions.
    """
    numbers = [r.number for r in plan.rooms]
    rng = np.random.default_rng(
        [seed, 4]
    )  # clutter uses [seed, 2], walkers [seed, 3]
    return [int(x) for x in rng.choice(numbers, size=n, replace=False)]


def markdown(rows: Sequence[Row]) -> str:
    """The table as Markdown, for the terminal and the plan log."""
    lines = [
        "| Condition | Missions | Success | Clean success | Median s "
        "| Collisions / mission |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        median = "--" if r.median_seconds is None else f"{r.median_seconds:.0f}"
        lines.append(
            f"| {r.condition} | {r.missions} | {r.success_rate:.0%} "
            f"| {r.clean_success_rate:.0%} | {median} "
            f"| {r.collisions_per_mission:.2f} |"
        )
    return "\n".join(lines) + "\n"
