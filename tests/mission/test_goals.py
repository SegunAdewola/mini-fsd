"""Tests for minifsd.mission.goals: "go to room 103" to a goal pose."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.mission import goals
from minifsd.world import floor_plan

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=2)


@pytest.mark.parametrize(
    "text",
    [
        "go to room 103",
        "Go to Room 103.",
        "room 103",
        "103",
        "  take me to room 103 please ",
        "Room #103",
    ],
)
def test_reads_the_room_number(text: str) -> None:
    assert goals.parse_room_number(text) == 103


@pytest.mark.parametrize(
    "text",
    ["", "go to the kitchen", "go to room 103 or 104", "room 1.5"],
)
def test_rejects_text_without_exactly_one_room_number(text: str) -> None:
    with pytest.raises(ValueError):
        goals.parse_room_number(text)


def test_goal_is_the_pose_in_front_of_that_door() -> None:
    room = PLAN.rooms[0]
    assert goals.parse_goal(f"go to room {room.number}", PLAN) == room.goal


def test_unknown_room_names_the_rooms_that_exist() -> None:
    with pytest.raises(goals.UnknownRoomError) as error:
        goals.parse_goal("go to room 999", PLAN)
    assert "999" in str(error.value)
    for room in PLAN.rooms:
        assert str(room.number) in str(error.value)


@settings(max_examples=30, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=6),
    st.integers(min_value=1, max_value=20),
)
def test_every_room_in_any_layout_resolves(
    seed: int, n: int, floor_number: int
) -> None:
    plan = floor_plan.generate_floor_plan(
        seed, rooms_per_side=n, floor_number=floor_number
    )
    for room in plan.rooms:
        assert goals.parse_goal(f"go to room {room.number}", plan) == room.goal


@given(st.integers(min_value=1, max_value=99_999))
def test_numbers_not_on_the_floor_raise(number: int) -> None:
    known = {room.number for room in PLAN.rooms}
    if number in known:
        return
    with pytest.raises(goals.UnknownRoomError):
        goals.parse_goal(f"room {number}", PLAN)
