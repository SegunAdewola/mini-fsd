"""Tests for minifsd.world.floor_plan.

The generator is checked with Hypothesis over many seeds: any seed must give a
closed hallway with open doors, unique room numbers and a clear spawn.
"""

import itertools
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.world import floor_plan

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)
ROOM_COUNTS = st.integers(min_value=1, max_value=8)
EPS = 1e-6  # keeps sample points off exact wall edges


def _inside(wall: floor_plan.Wall, x: float, y: float) -> bool:
    return wall.x_min < x < wall.x_max and wall.y_min < y < wall.y_max


def _in_any_wall(b: floor_plan.FloorPlan, x: float, y: float) -> bool:
    return any(_inside(w, x, y) for w in b.walls)


def _distance(wall: floor_plan.Wall, x: float, y: float) -> float:
    """Distance from a point to a wall rectangle; 0 if the point is inside."""
    dx = max(wall.x_min - x, 0.0, x - wall.x_max)
    dy = max(wall.y_min - y, 0.0, y - wall.y_max)
    return math.hypot(dx, dy)


def _clearance(b: floor_plan.FloorPlan, x: float, y: float) -> float:
    return min(_distance(w, x, y) for w in b.walls)


def _wall_mid_y(b: floor_plan.FloorPlan, side: int) -> float:
    """y halfway through the thickness of one hallway wall."""
    return side * (b.hallway_width / 2 + floor_plan.WALL_THICKNESS / 2)


# --- split_wall (hole 1) ----------------------------------------------------


@pytest.mark.parametrize(
    ("gaps", "expected"),
    [
        ([], [(0, 10)]),
        ([(2, 3)], [(0, 2), (3, 10)]),
        ([(6, 7), (2, 3)], [(0, 2), (3, 6), (7, 10)]),  # gaps out of order
        ([(0, 3)], [(3, 10)]),  # gap at the start: no zero-length piece
        ([(8, 10)], [(0, 8)]),  # gap at the end
    ],
)
def test_split_wall_cases(
    gaps: list[tuple[float, float]], expected: list[tuple[float, float]]
) -> None:
    assert floor_plan.split_wall(0, 10, gaps) == expected


@st.composite
def wall_with_gaps(
    draw: st.DrawFn,
) -> tuple[float, float, list[tuple[float, float]]]:
    """A wall from 0 to 100 and up to 5 non-overlapping gaps inside it."""
    cuts = draw(
        st.lists(
            st.integers(min_value=0, max_value=100),
            unique=True,
            max_size=10,
        )
    )
    cuts.sort()
    # Pair the cuts into gaps; with an odd count the last cut is dropped.
    pairs = zip(cuts[::2], cuts[1::2], strict=False)
    gaps = [(float(a), float(b)) for a, b in pairs]
    order = draw(st.permutations(gaps))
    return 0.0, 100.0, list(order)


@given(wall_with_gaps())
def test_split_wall_pieces_and_gaps_cover_the_wall_exactly(
    case: tuple[float, float, list[tuple[float, float]]],
) -> None:
    start, end, gaps = case
    pieces = floor_plan.split_wall(start, end, gaps)
    # Pieces have positive length, come sorted and don't overlap.
    assert all(a < b for a, b in pieces)
    assert all(p[1] <= q[0] for p, q in itertools.pairwise(pieces))
    # Solid plus open adds up to the whole wall.
    solid = sum(b - a for a, b in pieces)
    open_ = sum(b - a for a, b in gaps)
    assert solid + open_ == pytest.approx(end - start)
    # No piece reaches into a gap.
    for a, b in pieces:
        for ga, gb in gaps:
            assert b <= ga or a >= gb


# --- room_number (hole 2) ---------------------------------------------------


@pytest.mark.parametrize(
    ("floor_number", "index", "side", "expected"),
    [
        (1, 0, floor_plan.LEFT, 101),
        (1, 0, floor_plan.RIGHT, 102),
        (1, 1, floor_plan.LEFT, 103),
        (1, 1, floor_plan.RIGHT, 104),
        (2, 3, floor_plan.RIGHT, 208),
    ],
)
def test_room_number_matches_odd_left_even_right(
    floor_number: int, index: int, side: int, expected: int
) -> None:
    assert floor_plan.room_number(floor_number, index, side) == expected


# --- generate_floor_plan: hand-written ----------------------------------------


@pytest.mark.parametrize("rooms_per_side", [0, -1, 50])
def test_rejects_rooms_per_side_out_of_range(rooms_per_side: int) -> None:
    with pytest.raises(ValueError):
        floor_plan.generate_floor_plan(0, rooms_per_side=rooms_per_side)


def test_rejects_floor_below_one() -> None:
    with pytest.raises(ValueError):
        floor_plan.generate_floor_plan(0, floor_number=0)


def test_floor_sets_hundreds_digit() -> None:
    b = floor_plan.generate_floor_plan(0, rooms_per_side=2, floor_number=2)
    assert sorted(r.number for r in b.rooms) == [201, 202, 203, 204]


# --- generate_floor_plan: properties ------------------------------------------


@given(SEEDS, ROOM_COUNTS)
def test_room_numbers_are_unique(seed: int, n: int) -> None:
    b = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    numbers = [r.number for r in b.rooms]
    assert len(numbers) == 2 * n
    assert len(set(numbers)) == len(numbers)


@given(SEEDS, ROOM_COUNTS)
def test_every_door_gap_is_open(seed: int, n: int) -> None:
    b = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    half_door = b.door_width / 2 - EPS
    for room in b.rooms:
        y = _wall_mid_y(b, room.side)
        for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
            x = room.door_x + frac * half_door
            assert not _in_any_wall(b, x, y), (room.number, x)


@given(SEEDS, ROOM_COUNTS)
def test_hallway_walls_are_closed_outside_doors(seed: int, n: int) -> None:
    b = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    half_door = b.door_width / 2
    for side in (floor_plan.LEFT, floor_plan.RIGHT):
        doors = [r.door_x for r in b.rooms if r.side == side]
        y = _wall_mid_y(b, side)
        # Centres of 200 equal slices. Sampling x = 0 or x = length exactly
        # would land on the seam where two walls touch, which _inside's
        # strict comparison counts as neither wall.
        for i in range(200):
            x = b.length * (i + 0.5) / 200
            if any(abs(x - d) <= half_door + EPS for d in doors):
                continue  # inside a door gap: checked by the test above
            assert _in_any_wall(b, x, y), (side, x)


@given(SEEDS, ROOM_COUNTS)
def test_spawn_is_clear_of_walls(seed: int, n: int) -> None:
    b = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    assert _clearance(b, b.spawn.x, b.spawn.y) >= 0.5


@given(SEEDS, ROOM_COUNTS)
def test_goals_are_in_the_hallway_and_clear(seed: int, n: int) -> None:
    b = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    for room in b.rooms:
        g = room.goal
        assert 0 < g.x < b.length
        assert abs(g.y) < b.hallway_width / 2
        assert _clearance(b, g.x, g.y) >= 0.3


@given(SEEDS, ROOM_COUNTS)
def test_same_seed_gives_same_floor_plan(seed: int, n: int) -> None:
    a = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    b = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    assert a == b
