"""Smoke tests for perception_node: camera in, obstacle cloud out.

A probe node plays the camera: it publishes the camera's static transform,
camera_info, a depth image of a wall 3 m ahead and an RGB frame. The
geometry and the pipeline are tested in tests/perception (no ROS); this
checks the wiring and frames, with source:=truth (sim depth) and
source:=model (the ONNX models, M4.6). The model test needs the exported
models in data/models and skips without them.
"""

import pathlib
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest
import rclpy
import rclpy.executors
import rclpy.node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import minifsd
from mini_fsd_perception import perception_node
from minifsd.geometry import frames

TIMEOUT = 5.0
K = frames.intrinsics_from_fovy(320, 240, 50.0)
OPTICAL = "camera_optical_frame"
# src/minifsd/__init__.py -> the repo root (an editable install).
MODELS = pathlib.Path(minifsd.__file__).resolve().parents[2] / "data/models"
MODEL_TIMEOUT = 30.0  # loading three ONNX models takes a few seconds


class Camera(rclpy.node.Node):
    """Publishes what sim_node would, for a robot facing a wall 3 m ahead."""

    def __init__(self) -> None:
        super().__init__("fake_camera")
        self.clouds: list[PointCloud2] = []
        self.overlays: list[Image] = []
        self.create_subscription(
            PointCloud2, "camera/obstacles", self.clouds.append, 10
        )
        self.create_subscription(
            Image, "perception/overlay", self.overlays.append, 10
        )
        self.depth = self.create_publisher(Image, "camera/depth/image_raw", 10)
        self.info = self.create_publisher(
            CameraInfo, "camera/depth/camera_info", 10
        )
        self.rgb = self.create_publisher(Image, "camera/color/image_raw", 10)
        self.rgb_info = self.create_publisher(
            CameraInfo, "camera/color/camera_info", 10
        )
        t = TransformStamped()
        t.header.frame_id = "base_link"
        t.child_frame_id = OPTICAL
        t.transform.translation.z = 0.5
        q = frames.OPTICAL_TO_BASE_QUAT
        t.transform.rotation.x, t.transform.rotation.y = q.x, q.y
        t.transform.rotation.z, t.transform.rotation.w = q.z, q.w
        self._tf = tf2_ros.StaticTransformBroadcaster(self)
        self._tf.sendTransform(t)

    def publish(self) -> None:
        header = Header(frame_id=OPTICAL)
        info = CameraInfo(header=header, width=K.width, height=K.height)
        info.k = K.matrix.flatten().tolist()
        self.info.publish(info)
        self.rgb_info.publish(info)
        grey = np.full((K.height, K.width, 3), 128, dtype=np.uint8)
        self.rgb.publish(
            Image(
                header=header,
                height=K.height,
                width=K.width,
                encoding="rgb8",
                step=K.width * 3,
                data=grey.tobytes(),
            )
        )
        wall = np.full((K.height, K.width), 3.0, dtype=np.float32)
        self.depth.publish(
            Image(
                header=header,
                height=K.height,
                width=K.width,
                encoding="32FC1",
                step=K.width * 4,
                data=wall.tobytes(),
            )
        )


class Harness:
    def __init__(self, source: str = "truth") -> None:
        self.node = perception_node.PerceptionNode(
            parameter_overrides=[
                Parameter("source", value=source),
                Parameter("models_dir", value=str(MODELS)),
            ]
        )
        self.camera = Camera()
        self.executor = rclpy.executors.SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor.add_node(self.camera)

    def spin_until(
        self, done: Callable[[], bool], timeout: float = TIMEOUT
    ) -> None:
        deadline = time.monotonic() + timeout
        while not done():
            assert time.monotonic() < deadline, "timed out"
            self.camera.publish()  # like a camera: keep sending frames
            for _ in range(5):
                self.executor.spin_once(timeout_sec=0.01)

    def close(self) -> None:
        self.executor.shutdown()
        self.node.destroy_node()
        self.camera.destroy_node()


@pytest.fixture
def h() -> Iterator[Harness]:
    rclpy.init()
    try:
        harness = Harness()
        yield harness
        harness.close()
    finally:
        rclpy.shutdown()


def test_publishes_the_wall_in_the_robot_frame(h: Harness) -> None:
    h.spin_until(lambda: bool(h.camera.clouds))
    cloud = h.camera.clouds[-1]
    assert cloud.header.frame_id == "base_link"
    xyz = point_cloud2.read_points_numpy(cloud, field_names=("x", "y", "z"))
    assert len(xyz) > 100
    np.testing.assert_allclose(xyz[:, 0], 3.0, atol=1e-4)  # 3 m ahead
    assert xyz[:, 2].min() >= 0.05  # nothing at floor level


@pytest.mark.skipif(
    not (MODELS / perception_node.FLOOR_MODEL).exists(),
    reason="needs the exported ONNX models in data/models (M4.4)",
)
def test_model_mode_turns_rgb_into_a_cloud_and_an_overlay() -> None:
    rclpy.init()
    try:
        h = Harness(source="model")
        h.spin_until(
            lambda: bool(h.camera.clouds and h.camera.overlays),
            timeout=MODEL_TIMEOUT,
        )
        assert h.camera.clouds[-1].header.frame_id == "base_link"
        overlay = h.camera.overlays[-1]
        assert (overlay.width, overlay.height) == (K.width, K.height)
        assert overlay.encoding == "rgb8"
        h.close()
    finally:
        rclpy.shutdown()
