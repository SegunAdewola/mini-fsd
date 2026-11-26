"""ROS 2 node: the camera to an obstacle point cloud for Nav2 (R10, M4.6).

The logic lives in minifsd (ADR 0003); this node converts messages. Two
sources, chosen by the `source` parameter:

- truth (R10): camera/depth/image_raw, the sim's exact depth in metres,
  straight into obstacle points (minifsd.perception.obstacles).
- model (M4.6): camera/color/image_raw through the picked models in ONNX
  Runtime (minifsd.perception.pipeline): floor mask, boxes and depth scaled
  on the floor each frame. It also publishes perception/overlay, the RGB
  frame with the floor tinted and the boxes drawn, for RViz.

Either way the cloud goes to camera/obstacles in base_link, which the
costmaps' obstacle layers mark. The camera's pose comes from TF and its
intrinsics from CameraInfo, so nothing here assumes the sim.

Model mode keeps only the newest frame (queue depth 1): a model slower than
the camera should skip frames, not fall behind on old ones.
"""

import pathlib

import numpy as np
import rclpy
import rclpy.executors
import rclpy.node
import rclpy.time
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from minifsd.demo import overlay
from minifsd.eval import depth_align
from minifsd.geometry import frames
from minifsd.perception import detect_onnx, obstacles, onnx_models, pipeline

SOURCES = ("truth", "model")
# Files under models_dir: the T5 picks, exported in M4.4.
FLOOR_MODEL = "floor-dinov2-s-v10.onnx"
DEPTH_MODEL = "depth-student.onnx"
DETECT_MODEL = "detect-yolo26n/best.onnx"
# The student's global fit on 300 v10 train frames (M4.6), for frames with
# too little floor in view: 1 / depth = 0.172 d + 0.543.
FALLBACK_SCALE = 0.172
FALLBACK_SHIFT = 0.543


class PerceptionNode(rclpy.node.Node):
    """Publishes the obstacle points the camera sees."""

    def __init__(self, **kwargs: object) -> None:
        """Sets up the source's subscriptions, the publishers and TF.

        Args:
            **kwargs: Passed to rclpy's Node (tests set parameters this way).

        Raises:
            ValueError: For an unknown source.
        """
        super().__init__("perception", **kwargs)  # type: ignore[arg-type]
        self._target = self.declare_parameter("target_frame", "base_link").value
        source = self.declare_parameter("source", "truth").value
        models = pathlib.Path(
            self.declare_parameter("models_dir", "data/models").value
        )
        threads = self.declare_parameter("threads", 2).value
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
        self._intrinsics: frames.Intrinsics | None = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._cloud = self.create_publisher(PointCloud2, "camera/obstacles", 10)
        if source == "truth":
            self.create_subscription(
                CameraInfo, "camera/depth/camera_info", self._on_info, 10
            )
            self.create_subscription(
                Image, "camera/depth/image_raw", self._on_depth, 10
            )
            return
        self._pipeline = pipeline.Pipeline(
            floor=onnx_models.OnnxFloor(models / FLOOR_MODEL, threads).predict,
            depth=onnx_models.OnnxDepth(models / DEPTH_MODEL, threads).predict,
            detect=detect_onnx.OnnxDetector(
                models / DETECT_MODEL, threads
            ).predict,
            fallback=depth_align.Fit(FALLBACK_SCALE, FALLBACK_SHIFT),
        )
        self._frames = 0
        self._from_floor = 0
        self._overlay = self.create_publisher(Image, "perception/overlay", 1)
        self.create_subscription(
            CameraInfo, "camera/color/camera_info", self._on_info, 10
        )
        self.create_subscription(
            Image, "camera/color/image_raw", self._on_rgb, 1
        )
        self.get_logger().info(f"running the models from {models}")

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

    def _camera_pose(
        self, frame_id: str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """(rotation, translation) of the optical frame in base_link."""
        try:
            # The camera is fixed on the robot, so the latest transform
            # (time 0) is right for every image.
            t = self._tf_buffer.lookup_transform(
                self._target, frame_id, rclpy.time.Time()
            ).transform
        except tf2_ros.TransformException as error:
            self.get_logger().warn(
                f"no transform yet: {error}", throttle_duration_sec=2.0
            )
            return None
        rotation = frames.quaternion_to_matrix(
            frames.Quaternion(
                t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
            )
        )
        translation = np.array(
            [t.translation.x, t.translation.y, t.translation.z]
        )
        return rotation, translation

    def _publish_cloud(self, stamp: object, points: np.ndarray) -> None:
        header = Header(stamp=stamp, frame_id=self._target)
        self._cloud.publish(point_cloud2.create_cloud_xyz32(header, points))

    def _on_depth(self, msg: Image) -> None:
        if self._intrinsics is None:
            return  # no camera_info yet
        if msg.encoding != "32FC1":
            self.get_logger().error(
                f"expected 32FC1 depth, got {msg.encoding}", once=True
            )
            return
        pose = self._camera_pose(msg.header.frame_id)
        if pose is None:
            return
        # Rows may be padded: step is bytes per row, 4 bytes per float.
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape(
            msg.height, msg.step // 4
        )[:, : msg.width]
        points = obstacles.obstacle_points(depth, self._intrinsics, *pose)
        self._publish_cloud(msg.header.stamp, points)

    def _on_rgb(self, msg: Image) -> None:
        if self._intrinsics is None:
            return  # no camera_info yet
        if msg.encoding != "rgb8":
            self.get_logger().error(
                f"expected rgb8, got {msg.encoding}", once=True
            )
            return
        pose = self._camera_pose(msg.header.frame_id)
        if pose is None:
            return
        # Rows may be padded: step is bytes per row, 3 bytes per pixel.
        rgb = (
            np.frombuffer(msg.data, dtype=np.uint8)
            .reshape(msg.height, msg.step)[:, : msg.width * 3]
            .reshape(msg.height, msg.width, 3)
        )
        seen = self._pipeline(rgb, self._intrinsics, *pose)
        self._publish_cloud(msg.header.stamp, seen.points)
        drawn = overlay.draw_boxes(
            overlay.draw_floor(rgb, seen.floor), seen.boxes
        )
        self._overlay.publish(
            Image(
                header=msg.header,
                height=msg.height,
                width=msg.width,
                encoding="rgb8",
                step=msg.width * 3,
                data=drawn.tobytes(),
            )
        )
        # How often the floor, not the fallback, set the depth scale.
        self._frames += 1
        self._from_floor += int(seen.metric.from_floor)
        self.get_logger().info(
            f"{self._frames} frames, scale from the floor in "
            f"{100 * self._from_floor / self._frames:.0f}%",
            throttle_duration_sec=10.0,
        )


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
