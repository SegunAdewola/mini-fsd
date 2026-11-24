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
