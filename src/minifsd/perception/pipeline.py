"""One camera frame to obstacle points for the costmap (M4.5).

The robot's perception, without ROS (ADR 0003). Per frame:

1. the floor model marks floor pixels, the depth model gives relative
   disparity, and the detector gives boxes;
2. the floor fixes the depth's scale for this frame (metric_depth);
3. the metric depth becomes obstacle points (obstacles.obstacle_points), and
   the floor drops out there by height, like any other surface;
4. each confident person or obstacle box adds a short column of points
   where its bottom edge meets the floor (minifsd.geometry.ground).

Step 4 is for what depth handles badly: people, and thin things such as
chair legs, which a 160 x 120 depth map can blur into the floor. A box
whose bottom edge is at or above the horizon can't be on the floor and
adds nothing. Doors and signs are for the mission, not the costmap.

The models are plain functions, so the same pipeline runs on ONNX Runtime
in the ROS node, on torch in a notebook, or on fakes in the tests.
"""

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

from minifsd.eval import depth_align, detection
from minifsd.geometry import frames, ground
from minifsd.perception import metric_depth, obstacles

Floor = Callable[[np.ndarray], np.ndarray]
Depth = Callable[[np.ndarray], np.ndarray]
Detect = Callable[[np.ndarray], list[detection.Detection]]

COSTMAP_CLASSES = ("person", "obstacle")
# YOLO26n's scores; a box below this is more often a false obstacle (which
# blocks a corridor) than a missed one (which depth may still catch).
MIN_SCORE = 0.4
EDGE_SAMPLES = 5  # points along a box's bottom edge
# Heights of each column, m: inside obstacles' MIN_HEIGHT to MAX_HEIGHT,
# so the costmap's obstacle layer marks them like depth points.
COLUMN_HEIGHTS = (0.1, 0.3, 0.5)


@dataclasses.dataclass(frozen=True)
class Seen:
    """What one frame showed.

    Attributes:
        floor: (H, W) bool floor mask.
        metric: The metric depth and where its scale came from.
        boxes: Every detection, for the mission and RViz.
        points: (N, 3) float32 obstacle points in base_link.
    """

    floor: np.ndarray
    metric: metric_depth.Metric
    boxes: list[detection.Detection]
    points: np.ndarray


def box_points(
    boxes: Sequence[detection.Detection],
    k: frames.Intrinsics,
    rotation: np.ndarray,
    translation: np.ndarray,
    min_score: float = MIN_SCORE,
) -> np.ndarray:
    """Points for the confident person and obstacle boxes, in base_link.

    Args:
        boxes: Detections in image pixels.
        k: The camera's intrinsics.
        rotation: (3, 3) optical frame to base_link (from TF).
        translation: (3,) the camera's position in base_link (from TF).
        min_score: Lowest score that counts.

    Returns:
        (N, 3) float32: for each box, EDGE_SAMPLES floor points along its
        bottom edge, each raised to every COLUMN_HEIGHTS. Points beyond
        obstacles.MAX_RANGE are dropped, as for depth.
    """
    columns = []
    for box in boxes:
        if box.category not in COSTMAP_CLASSES or box.score < min_score:
            continue
        u = np.linspace(box.x, box.x + box.w, EDGE_SAMPLES)
        edge = np.stack([u, np.full_like(u, box.y + box.h)], axis=1)
        feet = ground.floor_points(edge, k, rotation, translation)
        feet = feet[np.isfinite(feet).all(axis=1)]
        feet = feet[np.hypot(feet[:, 0], feet[:, 1]) <= obstacles.MAX_RANGE]
        for z in COLUMN_HEIGHTS:
            columns.append(feet + np.array([0.0, 0.0, z]))
    if not columns:
        return np.zeros((0, 3), np.float32)
    return np.concatenate(columns).astype(np.float32)


class Pipeline:
    """The three models and the geometry that joins them."""

    def __init__(
        self,
        floor: Floor,
        depth: Depth,
        detect: Detect,
        fallback: depth_align.Fit,
        min_score: float = MIN_SCORE,
    ) -> None:
        """Takes the models as functions of an RGB image.

        Args:
            floor: RGB to (H, W) bool floor mask.
            depth: RGB to (H, W) relative disparity (larger is closer).
            detect: RGB to detections in image pixels.
            fallback: Depth scale to use when the floor can't give one.
            min_score: Lowest box score that adds costmap points.
        """
        self._floor = floor
        self._depth = depth
        self._detect = detect
        self._fallback = fallback
        self._min_score = min_score

    def __call__(
        self,
        rgb: np.ndarray,
        k: frames.Intrinsics,
        rotation: np.ndarray,
        translation: np.ndarray,
    ) -> Seen:
        """Runs one frame.

        Args:
            rgb: (H, W, 3) uint8, the size k describes.
            k: The camera's intrinsics.
            rotation: (3, 3) optical frame to base_link (from TF).
            translation: (3,) the camera's position in base_link (from TF).

        Returns:
            The floor, the metric depth, the boxes and the obstacle points.
        """
        floor = self._floor(rgb)
        metric = metric_depth.metric_depth(
            self._depth(rgb), floor, k, rotation, translation, self._fallback
        )
        boxes = self._detect(rgb)
        points = np.concatenate(
            [
                obstacles.obstacle_points(
                    metric.depth, k, rotation, translation
                ),
                box_points(boxes, k, rotation, translation, self._min_score),
            ]
        )
        return Seen(floor, metric, boxes, points)
