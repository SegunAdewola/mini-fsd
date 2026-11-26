"""Tests for minifsd.world.people: walkers in the hallway (M4.7)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.world import clutter, floor_plan, people, sim

FAR = (-100.0, -100.0)  # a robot position that never gets in the way


def test_a_walker_paces_its_segment_and_comes_back() -> None:
    walker = people.Walker(x_start=3.0, x_end=7.0, y=0.5, speed=1.0, phase=0.0)
    crowd = people.Crowd([walker])
    xs = []
    for _ in range(80):  # 8 s: there (4 m) and back (4 m)
        crowd.advance(0.1, FAR)
        ((x, y),) = crowd.positions()
        assert 3.0 - 1e-9 <= x <= 7.0 + 1e-9 and y == 0.5
        xs.append(x)
    assert max(xs) == pytest.approx(7.0, abs=0.11)
    assert xs[-1] == pytest.approx(3.0, abs=1e-6)  # back at the start


@settings(deadline=None, max_examples=30)
@given(st.integers(0, 10_000))
def test_walkers_stay_in_the_hallway_clear_of_walls_spawn_and_boxes(
    seed: int,
) -> None:
    plan = floor_plan.generate_floor_plan(seed)
    boxes = clutter.generate_clutter(plan, 6, np.random.default_rng([seed, 2]))
    walkers = people.generate_walkers(
        plan, np.random.default_rng([seed, 3]), 2, boxes
    )
    half = plan.hallway_width / 2
    for w in walkers:
        assert abs(w.y) + people.RADIUS <= half - people.MARGIN + 1e-9
        assert w.x_start >= plan.spawn.x + people.SPAWN_GAP - 1e-9
        assert w.x_end + people.RADIUS <= plan.length - people.MARGIN + 1e-9
        for x in np.arange(w.x_start, w.x_end, 0.05):
            assert not any(
                clutter.contains(b, x, w.y, grow=people.RADIUS) for b in boxes
            )
    # One per side of the hallway's centre line, so they never meet.
    assert len({math.copysign(1.0, w.y) for w in walkers}) == len(walkers)


def test_the_same_seed_gives_the_same_walkers() -> None:
    plan = floor_plan.generate_floor_plan(4)
    a = people.generate_walkers(plan, np.random.default_rng(1), 2)
    b = people.generate_walkers(plan, np.random.default_rng(1), 2)
    assert a == b and len(a) == 2


@settings(deadline=None, max_examples=50)
@given(
    st.floats(0.0, 10.0),  # robot x
    st.floats(-1.5, 1.5),  # robot y
    st.floats(0.0, 8.0),  # the walker's starting place on its path
)
def test_a_walker_never_steps_closer_than_keep_away_to_the_robot(
    rx: float, ry: float, phase: float
) -> None:
    walker = people.Walker(
        x_start=2.0, x_end=8.0, y=0.5, speed=1.2, phase=phase
    )
    crowd = people.Crowd([walker])
    for _ in range(100):
        ((x, y),) = crowd.positions()
        before = math.hypot(x - rx, y - ry)
        crowd.advance(0.05, (rx, ry))
        ((x, y),) = crowd.positions()
        after = math.hypot(x - rx, y - ry)
        # It may stand where it is, but it doesn't close in past KEEP_AWAY.
        assert after >= min(before, people.KEEP_AWAY) - 1e-9


def test_walkers_move_in_the_sim_and_stand_on_the_floor() -> None:
    plan = floor_plan.generate_floor_plan(0)
    walker = people.Walker(x_start=4.0, x_end=8.0, y=0.4, speed=1.0, phase=0.0)
    s = sim.Sim(plan, walkers=[walker])
    try:
        start = s.walker_positions()[0].copy()
        s.step(1.0)
        end = s.walker_positions()[0]
        assert end[0] - start[0] == pytest.approx(1.0, abs=0.06)
        assert end[1] == pytest.approx(0.4)
        assert end[2] == pytest.approx(people.HEIGHT / 2)  # feet on the floor
    finally:
        s.close()
