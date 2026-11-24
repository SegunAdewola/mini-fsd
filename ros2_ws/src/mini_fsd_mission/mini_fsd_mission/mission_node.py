"""ROS 2 node that turns "go to room 103" into a Nav2 NavigateToPose goal.

The parsing and the room lookup live in minifsd.mission.goals (ADR 0003).
This node only moves messages:

- /mission (std_msgs/String) in: an instruction, e.g. "go to room 103".
- navigate_to_pose (Nav2 action) out: the pose in front of that room's door.
- /mission/status (std_msgs/String) out: "accepted: room 103",
  "succeeded: room 103", "failed: room 103 (...)" or "rejected: <reason>".

It builds the floor plan from the same seed and rooms_per_side as the sim,
so the room numbers match the world.

Try it with:
    ros2 topic pub --once /mission std_msgs/String "data: 'go to room 103'"
"""

import rclpy
import rclpy.action
import rclpy.executors
import rclpy.node
import rclpy.task
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String

from minifsd.geometry import frames
from minifsd.mission import goals
from minifsd.world import floor_plan

MAP_FRAME = "map"


class MissionNode(rclpy.node.Node):
    """Sends one Nav2 goal per instruction and reports how it went."""

    def __init__(self) -> None:
        """Builds the floor plan from parameters and starts listening."""
        super().__init__("mission")
        seed = self.declare_parameter("seed", 0).value
        rooms = self.declare_parameter("rooms_per_side", 4).value
        self._plan = floor_plan.generate_floor_plan(seed, rooms_per_side=rooms)
        self._nav = rclpy.action.ActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self._status = self.create_publisher(String, "mission/status", 10)
        self.create_subscription(String, "mission", self._on_mission, 10)
        numbers = ", ".join(str(r.number) for r in self._plan.rooms)
        self.get_logger().info(f"ready; rooms on this floor: {numbers}")

    def _report(self, text: str) -> None:
        self.get_logger().info(text)
        self._status.publish(String(data=text))

    def _on_mission(self, msg: String) -> None:
        try:
            number = goals.parse_room_number(msg.data)
            goal = goals.parse_goal(msg.data, self._plan)
        except (ValueError, goals.UnknownRoomError) as error:
            self._report(f"rejected: {error}")
            return
        if not self._nav.server_is_ready():
            self._report("rejected: Nav2 is not running yet")
            return

        request = NavigateToPose.Goal()
        request.pose = PoseStamped()
        request.pose.header.frame_id = MAP_FRAME
        # A zero stamp means "use the latest transform", which is what a goal
        # on a static map wants.
        request.pose.pose.position.x = goal.x
        request.pose.pose.position.y = goal.y
        q = frames.yaw_to_quaternion(goal.yaw)
        request.pose.pose.orientation = Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)
        # Nav2 preempts the current goal if a new one arrives, so a second
        # instruction simply replaces the first.
        sent = self._nav.send_goal_async(request)
        sent.add_done_callback(lambda f: self._on_goal_response(f, number))

    def _on_goal_response(self, future: rclpy.task.Future, number: int) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self._report(f"failed: room {number} (Nav2 rejected the goal)")
            return
        self._report(f"accepted: room {number}")
        handle.get_result_async().add_done_callback(
            lambda f: self._on_result(f, number)
        )

    def _on_result(self, future: rclpy.task.Future, number: int) -> None:
        result = future.result()
        if result is not None and result.status == GoalStatus.STATUS_SUCCEEDED:
            self._report(f"succeeded: room {number}")
        else:
            status = "no result" if result is None else result.status
            self._report(f"failed: room {number} (status {status})")


def main() -> None:
    """Entry point for `ros2 run mini_fsd_mission mission_node`."""
    rclpy.init()
    node = MissionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl-C, or `ros2 launch` stopping its nodes: the normal ways out.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
