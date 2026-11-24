"""ROS 2 node that runs the MuJoCo simulator (minifsd.world.sim.Sim).

All simulation logic lives in minifsd (ADR 0003). This node only moves numbers
between Sim and ROS messages:

- /cmd_vel (Twist) in: Sim.command(v, w).
- A wall-clock timer steps Sim, then publishes /clock, /odom, the
  odom -> base_link transform, and the camera topics.
- At start-up, /tf_static gets base_link -> camera_link -> camera_optical_frame.

This node owns simulation time. Its own timer runs on the wall clock, since
it can't wait on a clock it produces. Every other node runs with
use_sim_time:=true and follows /clock, so if rendering slows the sim down,
Nav2 slows down with it instead of seeing time jump.

Run it with: ros2 run mini_fsd_sim sim_node --ros-args -p seed:=0
"""

import numpy as np
import rclpy
import rclpy.executors
import rclpy.node
import tf2_ros
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Quaternion,
    TransformStamped,
    Twist,
    Vector3,
)
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from minifsd.geometry import frames
from minifsd.world import clutter as clutter_lib
from minifsd.world import floor_plan, mjcf, sim

# Frame names (REP 105). Nav2, RViz and the bringup config use these.
ODOM_FRAME = "odom"
BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_link"  # robot axes: x forward, z up
OPTICAL_FRAME = "camera_optical_frame"  # image axes: x right, y down, z ahead


def _stamp(seconds: float) -> Time:
    sec = int(seconds)
    return Time(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


def _ros_quaternion(q: frames.Quaternion) -> Quaternion:
    return Quaternion(x=q.x, y=q.y, z=q.z, w=q.w)


def _camera_info(header: Header, k: frames.Intrinsics) -> CameraInfo:
    """CameraInfo for an ideal pinhole camera, with no distortion."""
    return CameraInfo(
        header=header,
        width=k.width,
        height=k.height,
        distortion_model="plumb_bob",
        d=[0.0] * 5,
        k=k.matrix.flatten().tolist(),
        r=np.eye(3).flatten().tolist(),
        # P is K with a zero fourth column (one camera, no stereo baseline).
        p=np.hstack([k.matrix, np.zeros((3, 1))]).flatten().tolist(),
    )


class SimNode(rclpy.node.Node):
    """Runs one floor plan and publishes what the robot senses."""

    def __init__(self, **kwargs: object) -> None:
        """Builds the floor plan from parameters and starts publishing.

        Args:
            **kwargs: Passed to rclpy's Node, e.g. parameter_overrides.
        """
        super().__init__("sim", **kwargs)  # type: ignore[arg-type]
        seed = self.declare_parameter("seed", 0).value
        rooms = self.declare_parameter("rooms_per_side", 4).value
        # Boxes in the world but not on Nav2's map (minifsd.world.clutter).
        n_clutter = self.declare_parameter("clutter", 0).value
        rate_hz = self.declare_parameter("rate_hz", 20.0).value
        self._dt = 1.0 / rate_hz

        plan = floor_plan.generate_floor_plan(seed, rooms_per_side=rooms)
        # Seeded from the floor plan's seed, so a rerun is the same world.
        self.clutter = clutter_lib.generate_clutter(
            plan, n_clutter, np.random.default_rng([seed, 2])
        )
        self._sim = sim.Sim(plan, clutter=self.clutter)
        self._intrinsics = frames.intrinsics_from_fovy(
            mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, mjcf.FOVY_DEG
        )

        self._clock_pub = self.create_publisher(Clock, "clock", 10)
        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._rgb_pub = self.create_publisher(
            Image, "camera/color/image_raw", 10
        )
        self._rgb_info_pub = self.create_publisher(
            CameraInfo, "camera/color/camera_info", 10
        )
        self._depth_pub = self.create_publisher(
            Image, "camera/depth/image_raw", 10
        )
        self._depth_info_pub = self.create_publisher(
            CameraInfo, "camera/depth/camera_info", 10
        )
        self._tf = tf2_ros.TransformBroadcaster(self)
        self._tf_static = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_static_frames()

        self.create_subscription(Twist, "cmd_vel", self._on_cmd_vel, 10)
        self.create_timer(self._dt, self._tick)
        self.get_logger().info(
            f"floor plan seed {seed}, {rooms} rooms per side, "
            f"{len(self.clutter)} clutter boxes, {rate_hz} Hz"
        )

    def _on_cmd_vel(self, msg: Twist) -> None:
        # A unicycle uses only forward speed and turn rate; the other four
        # Twist fields are ignored.
        self._sim.command(msg.linear.x, msg.angular.z)

    def _tick(self) -> None:
        self._sim.step(self._dt)
        stamp = _stamp(self._sim.time)
        self._clock_pub.publish(Clock(clock=stamp))
        self._publish_odom(stamp)
        self._publish_camera(stamp)

    def _publish_odom(self, stamp: Time) -> None:
        pose = self._sim.pose()  # true pose (ADR 0004)
        v, w = self._sim.velocity()
        q = _ros_quaternion(frames.yaw_to_quaternion(pose.yaw))

        odom = Odometry()
        odom.header = Header(stamp=stamp, frame_id=ODOM_FRAME)
        odom.child_frame_id = BASE_FRAME
        # The pose is in the header frame (odom); the twist is in the child
        # frame (base_link), so forward speed is twist.linear.x.
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.orientation = q
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self._odom_pub.publish(odom)

        t = TransformStamped()
        t.header = odom.header
        t.child_frame_id = BASE_FRAME
        t.transform.translation = Vector3(x=pose.x, y=pose.y, z=0.0)
        t.transform.rotation = q
        self._tf.sendTransform(t)

    def _publish_camera(self, stamp: Time) -> None:
        header = Header(stamp=stamp, frame_id=OPTICAL_FRAME)
        h, w = mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH

        rgb = self._sim.render_rgb()
        self._rgb_pub.publish(
            Image(
                header=header,
                height=h,
                width=w,
                encoding="rgb8",
                is_bigendian=0,
                step=w * 3,  # bytes per row
                data=rgb.tobytes(),
            )
        )
        depth = self._sim.render_depth()
        self._depth_pub.publish(
            Image(
                header=header,
                height=h,
                width=w,
                encoding="32FC1",  # one float32 per pixel, metres
                is_bigendian=0,
                step=w * 4,
                data=depth.tobytes(),
            )
        )
        info = _camera_info(header, self._intrinsics)
        self._rgb_info_pub.publish(info)
        self._depth_info_pub.publish(info)

    def _publish_static_frames(self) -> None:
        # The stamp doesn't matter for static frames: TF treats them as true
        # at all times.
        camera = TransformStamped()
        camera.header.frame_id = BASE_FRAME
        camera.child_frame_id = CAMERA_FRAME
        camera.transform.translation.z = mjcf.CAMERA_HEIGHT
        camera.transform.rotation.w = 1.0  # same axes as the robot

        optical = TransformStamped()
        optical.header.frame_id = CAMERA_FRAME
        optical.child_frame_id = OPTICAL_FRAME
        optical.transform.rotation = _ros_quaternion(
            frames.OPTICAL_TO_BASE_QUAT
        )
        self._tf_static.sendTransform([camera, optical])

    def destroy_node(self) -> None:
        """Frees the renderer, then the node."""
        self._sim.close()
        super().destroy_node()


def main() -> None:
    """Entry point for `ros2 run mini_fsd_sim sim_node`."""
    rclpy.init()
    node = SimNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl-C, or `ros2 launch` stopping its nodes: the normal ways out.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
