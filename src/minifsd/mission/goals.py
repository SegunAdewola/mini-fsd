"""Turns a typed instruction such as "go to room 103" into a goal pose.

The language part is deliberately small: the instruction must contain exactly
one room number, and everything else in it is ignored. The goal is the pose
the floor plan stores for that room, 0.6 m in front of its door and facing it
(floor_plan.GOAL_STANDOFF).
"""

import re

from minifsd.world import floor_plan

# A whole number that is not part of a decimal: "103" and "103." match, but
# neither digit in "1.5" does.
_NUMBER = re.compile(r"(?<![\d.])\d+(?!\.?\d)")


class UnknownRoomError(LookupError):
    """The instruction names a room that isn't on this floor."""


def parse_room_number(text: str) -> int:
    """Finds the one room number in an instruction.

    Args:
        text: For example "go to room 103" or just "103".

    Returns:
        The room number.

    Raises:
        ValueError: If the text has no whole number, or more than one.
    """
    numbers = _NUMBER.findall(text)
    if len(numbers) != 1:
        raise ValueError(
            f"expected exactly one room number in {text!r}, found {numbers}"
        )
    return int(numbers[0])


def parse_goal(text: str, plan: floor_plan.FloorPlan) -> floor_plan.Pose2D:
    """The goal pose for the room an instruction names.

    Args:
        text: The instruction.
        plan: The floor the robot is on.

    Returns:
        The pose in front of that room's door, in the map frame.

    Raises:
        ValueError: If the text doesn't hold exactly one room number.
        UnknownRoomError: If no room on the floor has that number.
    """
    number = parse_room_number(text)
    for room in plan.rooms:
        if room.number == number:
            return room.goal
    known = ", ".join(str(r.number) for r in sorted(plan.rooms, key=_number))
    raise UnknownRoomError(f"no room {number} on this floor; rooms: {known}")


def _number(room: floor_plan.Room) -> int:
    return room.number
