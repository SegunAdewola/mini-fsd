"""Smoke tests for mission_node against a fake NavigateToPose server.

The fake stands in for Nav2's bt_navigator, so these tests check only our
node: the instruction becomes the right goal pose, and the outcome comes back
on /mission/status. Driving to the goal is the hand test (R9 in the plan).
"""

import math
import time
from collections.abc import Callable, Iterator

import pytest
import rclpy
import rclpy.action
import rclpy.executors
import rclpy.node
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

from mini_fsd_mission import mission_node
from minifsd.geometry import frames
from minifsd.world import floor_plan

TIMEOUT = 5.0
# The node's defaults; the plan here must be the one the node builds.
PLAN = floor_plan.generate_floor_plan(0, rooms_per_side=4)


class FakeNav2(rclpy.node.Node):
    """Accepts every goal, records it, and reports success."""

    def __init__(self) -> None:
        super().__init__("fake_nav2")
        self.goals: list[NavigateToPose.Goal] = []
        self.status: list[str] = []
        self._server = rclpy.action.ActionServer(
            self, NavigateToPose, "navigate_to_pose", self._execute
        )
        self.create_subscription(
            String, "mission/status", lambda m: self.status.append(m.data), 10
        )
        self.mission = self.create_publisher(String, "mission", 10)

    def _execute(self, handle):  # handle: ServerGoalHandle
        self.goals.append(handle.request)
        handle.succeed()
        return NavigateToPose.Result()


class Harness:
    def __init__(self) -> None:
        self.node = mission_node.MissionNode()
        self.nav2 = FakeNav2()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.nav2)
        # Publishing before the subscriber is matched would lose the message.
        self.spin_until(
            lambda: (
                self.nav2.mission.get_subscription_count() > 0
                and self.node.count_subscribers("mission/status") > 0
            )
        )

    def spin_until(self, done: Callable[[], bool]) -> None:
        deadline = time.monotonic() + TIMEOUT
        while not done():
            assert time.monotonic() < deadline, "timed out"
            self.executor.spin_once(timeout_sec=0.01)

    def send(self, text: str) -> None:
        self.nav2.mission.publish(String(data=text))

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        self.nav2.destroy_node()


@pytest.fixture
def h() -> Iterator[Harness]:
    rclpy.init()
    try:  # shut down even if setup fails, or every later test fails too
        harness = Harness()
        yield harness
        harness.close()
    finally:
        rclpy.shutdown()


def test_sends_the_pose_in_front_of_the_door(h: Harness) -> None:
    room = PLAN.rooms[2]
    h.send(f"go to room {room.number}")
    h.spin_until(lambda: bool(h.nav2.goals))
    pose = h.nav2.goals[0].pose
    assert pose.header.frame_id == "map"
    p, q = pose.pose.position, pose.pose.orientation
    assert (p.x, p.y) == pytest.approx((room.goal.x, room.goal.y))
    yaw = frames.quaternion_to_yaw(frames.Quaternion(q.x, q.y, q.z, q.w))
    assert math.remainder(yaw - room.goal.yaw, 2 * math.pi) == pytest.approx(
        0.0, abs=1e-9
    )


def test_reports_success(h: Harness) -> None:
    room = PLAN.rooms[0]
    h.send(f"room {room.number}")
    h.spin_until(lambda: f"succeeded: room {room.number}" in h.nav2.status)
    assert h.nav2.status[0] == f"accepted: room {room.number}"


def test_rejects_an_unknown_room_without_moving(h: Harness) -> None:
    h.send("go to room 999")
    h.spin_until(lambda: bool(h.nav2.status))
    assert h.nav2.status[0].startswith("rejected: no room 999")
    assert not h.nav2.goals


def test_rejects_text_without_a_room_number(h: Harness) -> None:
    h.send("go to the kitchen")
    h.spin_until(lambda: bool(h.nav2.status))
    assert h.nav2.status[0].startswith("rejected:")
    assert not h.nav2.goals
