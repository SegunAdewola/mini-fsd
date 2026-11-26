"""Tests for minifsd.eval.closed_loop: counting bumps and the M4.8 table."""

import pathlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.eval import closed_loop
from minifsd.world import floor_plan


def test_one_long_bump_is_one_collision() -> None:
    bumps = closed_loop.Bumps()
    for contacts in (0, 0, 2, 3, 3, 1, 0, 0):
        bumps.update(contacts)
    assert bumps.total == 1


def test_each_new_touch_after_a_gap_counts_again() -> None:
    bumps = closed_loop.Bumps()
    for contacts in (1, 0, 1, 1, 0, 0, 4):
        bumps.update(contacts)
    assert bumps.total == 3


@given(st.lists(st.integers(0, 5), max_size=60))
def test_bumps_are_the_starts_of_runs_of_contact(contacts: list[int]) -> None:
    bumps = closed_loop.Bumps()
    for c in contacts:
        bumps.update(c)
    touching = [c > 0 for c in contacts]
    starts = sum(
        1
        for i, now in enumerate(touching)
        if now and (i == 0 or not touching[i - 1])
    )
    assert bumps.total == starts


def _record(
    condition: str, outcome: str, seconds: float, collisions: int
) -> closed_loop.Mission:
    return closed_loop.Mission(condition, 0, 101, outcome, seconds, collisions)


RECORDS = [
    _record("truth", "succeeded", 30.0, 0),
    _record("truth", "succeeded", 40.0, 0),
    _record("truth", "succeeded", 50.0, 1),
    _record("truth", "failed", 12.0, 0),
    _record("model", "succeeded", 45.0, 2),
    _record("model", "timeout", 120.0, 1),
]


def test_the_table_has_one_row_per_condition() -> None:
    rows = {r.condition: r for r in closed_loop.summarize(RECORDS)}
    truth, model = rows["truth"], rows["model"]
    assert (truth.missions, model.missions) == (4, 2)
    assert truth.success_rate == pytest.approx(0.75)
    # Succeeded without touching anything.
    assert truth.clean_success_rate == pytest.approx(0.5)
    assert truth.median_seconds == pytest.approx(40.0)  # successes only
    assert truth.collisions_per_mission == pytest.approx(0.25)
    assert model.success_rate == pytest.approx(0.5)
    assert model.clean_success_rate == 0.0
    assert model.collisions_per_mission == pytest.approx(1.5)


def test_a_condition_with_no_success_has_no_median_time() -> None:
    (row,) = closed_loop.summarize([_record("model", "timeout", 120.0, 0)])
    assert row.median_seconds is None


def test_records_round_trip_through_jsonl(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "results.jsonl"
    for r in RECORDS:
        closed_loop.append(r, path)
    assert closed_loop.load(path) == RECORDS


def test_rooms_are_distinct_from_the_plan_and_the_same_each_time() -> None:
    plan = floor_plan.generate_floor_plan(3)
    rooms = closed_loop.pick_rooms(plan, 4, seed=3)
    assert len(set(rooms)) == 4
    assert set(rooms) <= {r.number for r in plan.rooms}
    assert closed_loop.pick_rooms(plan, 4, seed=3) == rooms


def test_the_markdown_table_has_a_line_per_condition() -> None:
    text = closed_loop.markdown(closed_loop.summarize(RECORDS))
    lines = text.strip().splitlines()
    assert lines[0].startswith("| Condition")
    assert len(lines) == 2 + 2  # header, rule, truth, model
    assert "| truth | 4 | 75% | 50% | 40 | 0.25 |" in lines
    assert "| model | 2 | 50% | 0% | 45 | 1.50 |" in lines


@pytest.mark.parametrize(
    ("lines", "outcome"),
    [
        (["accepted: room 105"], None),  # still driving
        (["accepted: room 105", "succeeded: room 105"], "succeeded"),
        (["failed: room 105 (status 6)"], "failed"),
        (["rejected: Nav2 is not running yet"], "failed"),
        # The goal this one replaced reports its own end: not ours.
        (["accepted: room 105", "failed: room 106 (status 6)"], None),
        (["failed: room 1050 (status 6)"], None),  # a different room
    ],
)
def test_only_the_current_rooms_status_ends_a_mission(
    lines: list[str], outcome: str | None
) -> None:
    assert closed_loop.outcome_of(lines, 105) == outcome
