"""Tests for minifsd.world.clutter: furniture-sized boxes off the map."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.world import clutter, floor_plan

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=4)


def _corners(
    box: clutter.Box, shrink: float = 0.0
) -> list[tuple[float, float]]:
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            dx, dy = sx * (box.half_x - shrink), sy * (box.half_y - shrink)
            out.append((box.x + c * dx - s * dy, box.y + s * dx + c * dy))
    return out


def _in_wall(plan: floor_plan.FloorPlan, x: float, y: float) -> bool:
    return any(
        w.x_min < x < w.x_max and w.y_min < y < w.y_max for w in plan.walls
    )


@settings(max_examples=25, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=0, max_value=10),
)
def test_clutter_never_cuts_off_a_room(seed: int, rooms: int, n: int) -> None:
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=rooms)
    boxes = clutter.generate_clutter(plan, n, np.random.default_rng(seed))
    assert len(boxes) <= n
    assert clutter.reachable(plan, boxes)
    for box in boxes:
        # A cell of slack: placement checks on the 5 cm grid.
        for x, y in [(box.x, box.y), *_corners(box, shrink=0.05)]:
            assert not _in_wall(plan, x, y), box
        spawn = plan.spawn
        assert not clutter.contains(
            box, spawn.x, spawn.y, clutter.SPAWN_CLEARANCE
        )


def test_usually_places_every_box() -> None:
    boxes = clutter.generate_clutter(PLAN, 8, np.random.default_rng(0))
    assert len(boxes) == 8
    for b in boxes:
        assert clutter.MIN_HALF <= b.half_x <= clutter.MAX_HALF
        assert clutter.MIN_HEIGHT <= b.height <= clutter.MAX_HEIGHT


def test_is_reproducible() -> None:
    a = clutter.generate_clutter(PLAN, 5, np.random.default_rng(4))
    b = clutter.generate_clutter(PLAN, 5, np.random.default_rng(4))
    assert a == b


def test_no_clutter_is_reachable() -> None:
    assert clutter.generate_clutter(PLAN, 0, np.random.default_rng(0)) == ()
    assert clutter.reachable(PLAN, ())


def test_a_box_across_the_hallway_is_not_reachable() -> None:
    # A wall-to-wall box just past the first door cuts off every room
    # behind it.
    door = sorted(r.door_x for r in PLAN.rooms)[1]
    block = clutter.Box(
        x=door - 0.9,
        y=0.0,
        yaw=0.0,
        half_x=0.2,
        half_y=PLAN.hallway_width / 2,
        height=1.0,
        rgba=(0.5, 0.5, 0.5, 1.0),
    )
    assert not clutter.reachable(PLAN, (block,))


def test_contains_handles_rotation() -> None:
    box = clutter.Box(0.0, 0.0, math.pi / 2, 1.0, 0.2, 0.5, (0, 0, 0, 1))
    # Rotated 90 degrees: long along y, thin along x.
    assert clutter.contains(box, 0.0, 0.9)
    assert not clutter.contains(box, 0.9, 0.0)
    assert clutter.contains(box, 0.3, 0.0, grow=0.15)


# generate_clutter(plan 3, 3 rooms a side, 6 boxes, rng [3, 2]) before D11,
# rounded to 6 places: plain boxes must stay exactly what v9 and the ROS
# world were built with.
BEFORE_D11 = [
    (2.725, 3.925, 1.528746, 0.516049, 0.206806, 0.300079),
    (9.375, 3.125, -0.261804, 0.293147, 0.163519, 0.428279),
    (8.125, -3.475, 2.95597, 0.28018, 0.406519, 0.836867),
    (2.425, 2.125, -1.518987, 0.317938, 0.510902, 0.30653),
    (1.975, -2.725, 1.015912, 0.328784, 0.366831, 0.523144),
    (4.275, 2.375, -0.615059, 0.396716, 0.406704, 0.828412),
]


def test_default_clutter_is_what_it_was_before_furniture() -> None:
    plan = floor_plan.generate_floor_plan(3, rooms_per_side=3)
    boxes = clutter.generate_clutter(plan, 6, np.random.default_rng([3, 2]))
    got = [
        tuple(
            round(v, 6) for v in (b.x, b.y, b.yaw, b.half_x, b.half_y, b.height)
        )
        for b in boxes
    ]
    assert got == BEFORE_D11
    assert all(b.kind == "box" and b.surface == -1 for b in boxes)


@settings(max_examples=25, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=5),
)
def test_furniture_stays_in_range_and_never_cuts_off_a_room(
    seed: int, rooms: int
) -> None:
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=rooms)
    boxes = clutter.generate_clutter(
        plan,
        10,
        np.random.default_rng(seed),
        kinds=clutter.FURNITURE_KINDS,
        surfaces=3,
    )
    assert clutter.reachable(plan, boxes)
    for b in boxes:
        assert b.kind in clutter.FURNITURE_KINDS
        assert 0 <= b.surface < 3
        low, high = clutter.HEIGHTS[b.kind]
        assert low <= b.height <= high
        low, high = clutter.HALVES[b.kind]
        assert low <= b.half_x <= high and low <= b.half_y <= high


def test_every_furniture_kind_turns_up() -> None:
    kinds = {
        b.kind
        for seed in range(5)
        for b in clutter.generate_clutter(
            PLAN,
            10,
            np.random.default_rng(seed),
            kinds=clutter.FURNITURE_KINDS,
            surfaces=2,
        )
    }
    assert kinds == set(clutter.FURNITURE_KINDS)


def test_unknown_furniture_is_refused() -> None:
    import pytest

    with pytest.raises(ValueError):
        clutter.generate_clutter(
            PLAN, 3, np.random.default_rng(0), kinds=("sofa",)
        )
