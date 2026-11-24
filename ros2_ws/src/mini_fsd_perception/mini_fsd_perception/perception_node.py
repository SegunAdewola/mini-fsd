"""ROS 2 node: depth image to an obstacle point cloud for Nav2 (R10).

The geometry lives in minifsd.perception.obstacles (ADR 0003). This node:

- reads camera/depth/camera_info once for the intrinsics;
- for each camera/depth/image_raw (32FC1, metres), looks up the camera's pose
  in base_link from TF and turns the depth into obstacle points;
- publishes them on camera/obstacles (PointCloud2, in base_link), which the
  local costmap's obstacle layer marks.

Nothing here assumes the sim: the intrinsics come from CameraInfo and the
camera pose from TF, so M4 can feed it a model's depth from any camera.
"""

import numpy as np
import rclpy
import rclpy.executors
import rclpy.node
import rclpy.time
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from minifsd.geometry import frames
from minifsd.perception import obstacles


class PerceptionNode(rclpy.node.Node):
    """Publishes the obstacle points of every depth image."""

    def __init__(self) -> None:
        """Sets up the subscriptions, the publisher and the TF listener."""
        super().__init__("perception")
        self._target = self.declare_parameter("target_frame", "base_link").value
        self._intrinsics: frames.Intrinsics | None = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._cloud = self.create_publisher(PointCloud2, "camera/obstacles", 10)
        self.create_subscription(
            CameraInfo, "camera/depth/camera_info", self._on_info, 10
        )
        self.create_subscription(
            Image, "camera/depth/image_raw", self._on_depth, 10
        )

    def _on_info(self, msg: CameraInfo) -> None:
        # K is row-major: [fx 0 cx; 0 fy cy; 0 0 1].
        k = msg.k
        self._intrinsics = frames.Intrinsics(
            width=msg.width,
            height=msg.height,
            fx=k[0],
            fy=k[4],
            cx=k[2],
            cy=k[5],
        )

    def _on_depth(self, msg: Image) -> None:
        if self._intrinsics is None:
            return  # no camera_info yet
        if msg.encoding != "32FC1":
            self.get_logger().error(
                f"expected 32FC1 depth, got {msg.encoding}", once=True
            )
            return
        try:
            # The camera is fixed on the robot, so the latest transform
            # (time 0) is right for every image.
            t = self._tf_buffer.lookup_transform(
                self._target, msg.header.frame_id, rclpy.time.Time()
            ).transform
        except tf2_ros.TransformException as error:
            self.get_logger().warn(
                f"no transform yet: {error}", throttle_duration_sec=2.0
            )
            return
        rotation = frames.quaternion_to_matrix(
            frames.Quaternion(
                t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
            )
        )
        translation = np.array(
            [t.translation.x, t.translation.y, t.translation.z]
        )
        # Rows may be padded: step is bytes per row, 4 bytes per float.
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.step // 4
        )[:, : msg.width]
        points = obstacles.obstacle_points(
            depth, self._intrinsics, rotation, translation
        )
        header = Header(stamp=msg.header.stamp, frame_id=self._target)
        self._cloud.publish(point_cloud2.create_cloud_xyz32(header, points))


def main() -> None:
    """Entry point for `ros2 run mini_fsd_perception perception_node`."""
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        # Ctrl-C, or `ros2 launch` stopping its nodes: the normal ways out.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
