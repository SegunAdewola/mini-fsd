"""Smoke tests for sim_node: the ROS wiring around minifsd.world.sim.Sim.

The simulator's behaviour is tested in tests/world (no ROS). These tests only
check that the node publishes the right topics and frames and that /cmd_vel
reaches the simulator. The sim node and a probe node share one executor in
this process, so no launch file or second terminal is needed.

The node renders camera images, so these tests need a GPU, like the gl tests.
Run them with `pixi run test-ros`.
"""

import math
import time
from collections.abc import Callable, Iterator

import pytest
import rclpy
import rclpy.executors
import rclpy.node
import rclpy.parameter
import rclpy.time
import tf2_ros
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image

from mini_fsd_sim import sim_node
from minifsd.geometry import frames
from minifsd.world import mjcf

TIMEOUT = 5.0  # wall-clock seconds to wait for anything


class Probe(rclpy.node.Node):
    """Records the latest message on each topic the sim publishes."""

    def __init__(self) -> None:
        super().__init__("probe")
        self.odom: list[Odometry] = []
        self.clock: list[Clock] = []
        self.rgb: list[Image] = []
        self.depth: list[Image] = []
        self.info: list[CameraInfo] = []
        self.create_subscription(Odometry, "odom", self.odom.append, 10)
        self.create_subscription(Clock, "clock", self.clock.append, 10)
        self.create_subscription(
            Image, "camera/color/image_raw", self.rgb.append, 10
        )
        self.create_subscription(
            Image, "camera/depth/image_raw", self.depth.append, 10
        )
        self.create_subscription(
            CameraInfo, "camera/color/camera_info", self.info.append, 10
        )
        self.cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


class Harness:
    def __init__(self) -> None:
        self.sim = sim_node.SimNode()
        self.probe = Probe()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.sim)
        self.executor.add_node(self.probe)

    def spin_until(self, done: Callable[[], bool]) -> None:
        deadline = time.monotonic() + TIMEOUT
        while not done():
            assert time.monotonic() < deadline, "timed out"
            self.executor.spin_once(timeout_sec=0.01)

    def close(self) -> None:
        self.executor.shutdown()
        self.sim.destroy_node()
        self.probe.destroy_node()


@pytest.fixture
def h() -> Iterator[Harness]:
    rclpy.init()
    harness = Harness()
    yield harness
    harness.close()
    rclpy.shutdown()


def _seconds(stamp: object) -> float:
    return stamp.sec + stamp.nanosec * 1e-9  # type: ignore[attr-defined]


def test_publishes_odom_in_the_odom_frame(h: Harness) -> None:
    h.spin_until(lambda: bool(h.probe.odom))
    odom = h.probe.odom[-1]
    assert odom.header.frame_id == "odom"
    assert odom.child_frame_id == "base_link"


def test_clock_advances(h: Harness) -> None:
    h.spin_until(lambda: len(h.probe.clock) >= 2)
    first, last = h.probe.clock[0], h.probe.clock[-1]
    assert _seconds(last.clock) > _seconds(first.clock)


def test_camera_images_are_in_the_optical_frame(h: Harness) -> None:
    h.spin_until(lambda: bool(h.probe.rgb and h.probe.depth and h.probe.info))
    rgb, depth, info = h.probe.rgb[-1], h.probe.depth[-1], h.probe.info[-1]
    for msg in (rgb, depth, info):
        assert msg.header.frame_id == "camera_optical_frame"
    assert (rgb.width, rgb.height) == (mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT)
    assert rgb.encoding == "rgb8"
    assert len(rgb.data) == mjcf.IMAGE_WIDTH * mjcf.IMAGE_HEIGHT * 3
    assert depth.encoding == "32FC1"
    assert len(depth.data) == mjcf.IMAGE_WIDTH * mjcf.IMAGE_HEIGHT * 4
    k = frames.intrinsics_from_fovy(
        mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, mjcf.FOVY_DEG
    )
    assert list(info.k) == pytest.approx(k.matrix.flatten().tolist())


def test_static_frames_put_the_camera_on_the_robot(h: Harness) -> None:
    def found() -> bool:
        return h.probe.tf_buffer.can_transform(
            "base_link", "camera_optical_frame", rclpy.time.Time()
        )

    h.spin_until(found)
    t = h.probe.tf_buffer.lookup_transform(
        "base_link", "camera_optical_frame", rclpy.time.Time()
    ).transform
    assert (t.translation.x, t.translation.y, t.translation.z) == pytest.approx(
        (0.0, 0.0, mjcf.CAMERA_HEIGHT)
    )
    q = (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    # TF composes the two static transforms and may return -q, which is the
    # same rotation as q, so compare up to sign.
    expected = tuple(frames.OPTICAL_TO_BASE_QUAT)
    flipped = tuple(-c for c in expected)
    assert q == pytest.approx(expected) or q == pytest.approx(flipped)


def test_forward_command_moves_odom(h: Harness) -> None:
    h.spin_until(lambda: bool(h.probe.odom))
    start = h.probe.odom[-1]
    t0 = _seconds(start.header.stamp)

    def driven_one_second() -> bool:
        h.probe.cmd_vel.publish(Twist(linear=Vector3(x=0.5)))
        last = h.probe.odom[-1]
        return _seconds(last.header.stamp) - t0 >= 1.0

    h.spin_until(driven_one_second)
    end = h.probe.odom[-1]
    dx = end.pose.pose.position.x - start.pose.pose.position.x
    # 0.5 m/s for 1 s of sim time, less the actuator lag and the first tick
    # or two before the command arrives.
    assert 0.35 < dx < 0.55
    assert end.twist.twist.linear.x == pytest.approx(0.5, abs=0.05)
    assert math.isclose(end.pose.pose.position.y, 0.0, abs_tol=0.01)


def test_clutter_parameter_puts_boxes_in_the_world() -> None:
    rclpy.init()
    try:
        node = sim_node.SimNode(
            parameter_overrides=[rclpy.parameter.Parameter("clutter", value=6)]
        )
        assert len(node.clutter) == 6
        # The same seed gives the same boxes, so a rerun is the same world.
        again = sim_node.SimNode(
            parameter_overrides=[rclpy.parameter.Parameter("clutter", value=6)]
        )
        assert node.clutter == again.clutter
        node.destroy_node()
        again.destroy_node()
    finally:
        rclpy.shutdown()


def test_people_parameter_puts_walkers_in_the_world_clear_of_boxes() -> None:
    rclpy.init()
    try:
        node = sim_node.SimNode(
            parameter_overrides=[
                rclpy.parameter.Parameter("clutter", value=6),
                rclpy.parameter.Parameter("people", value=2),
            ]
        )
        # Boxes can crowd one lane out (seed 0's right half), so up to 2.
        assert 1 <= len(node.walkers) <= 2
        # Walkers draw from their own random stream, so asking for them
        # leaves the boxes as they were.
        boxes_only = sim_node.SimNode(
            parameter_overrides=[rclpy.parameter.Parameter("clutter", value=6)]
        )
        assert node.clutter == boxes_only.clutter
        assert boxes_only.walkers == ()
        node.destroy_node()
        boxes_only.destroy_node()
    finally:
        rclpy.shutdown()
