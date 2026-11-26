"""Tests for minifsd.perception.pipeline: one camera frame to obstacles (M4.5).

The models are fakes that return a scene's truth, so these test the joining
(scale, projection, boxes), not the models.
"""

from collections.abc import Callable

import numpy as np
import pytest

from minifsd.eval import depth_align, detection
from minifsd.geometry import frames
from minifsd.perception import obstacles, pipeline

K = frames.intrinsics_from_fovy(320, 240, 50.0)  # as in conftest.py
FALLBACK = depth_align.Fit(scale=0.5, shift=0.1)
H = 0.5  # camera height, m
RGB = np.zeros((K.height, K.width, 3), dtype=np.uint8)  # the fakes ignore it

Room = Callable[..., tuple[np.ndarray, np.ndarray]]
Pose = Callable[..., tuple[np.ndarray, np.ndarray]]


def _box(
    category: str, bottom: float, score: float = 0.9
) -> detection.Detection:
    return detection.Detection(
        category, 140.0, bottom - 60.0, 40.0, 60.0, score
    )


def _run(
    depth: np.ndarray,
    floor: np.ndarray,
    boxes: list[detection.Detection],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> pipeline.Seen:
    see = pipeline.Pipeline(
        floor=lambda rgb: floor,
        depth=lambda rgb: 2.0 / depth + 0.3,  # relative: unknown scale, shift
        detect=lambda rgb: boxes,
        fallback=FALLBACK,
    )
    return see(RGB, K, rotation, translation)


def test_a_box_becomes_points_where_its_bottom_meets_the_floor(
    make_pose: Pose,
) -> None:
    rotation, translation = make_pose(H, 0.0)
    box = _box("person", bottom=200.0)
    points = pipeline.box_points([box], K, rotation, translation)
    assert len(points) > 0
    # A level camera at H sees the floor at row v at H fy / (v - cy) ahead.
    ahead = H * K.fy / (200.0 - K.cy)
    np.testing.assert_allclose(points[:, 0], ahead, rtol=1e-6)
    # Across the box's width: columns 140 to 180 lie left of centre (159.5)
    # to right of it, so base y (left is positive) spans both signs.
    assert points[:, 1].min() < 0.0 < points[:, 1].max()
    # float32 points, so compare at float32's precision.
    levels = sorted({round(float(z), 5) for z in points[:, 2]})
    assert levels == pytest.approx(pipeline.COLUMN_HEIGHTS, abs=1e-5)
    heights = points[:, 2]
    assert heights.min() >= obstacles.MIN_HEIGHT
    assert heights.max() <= obstacles.MAX_HEIGHT


@pytest.mark.parametrize(
    "box",
    [
        _box("door", bottom=200.0),
        _box("room_sign", bottom=200.0),
        _box("person", bottom=200.0, score=0.1),  # too unsure
        _box("person", bottom=100.0),  # bottom above the horizon
    ],
)
def test_doors_signs_weak_boxes_and_floating_boxes_add_no_points(
    make_pose: Pose, box: detection.Detection
) -> None:
    rotation, translation = make_pose(H, 0.0)
    assert len(pipeline.box_points([box], K, rotation, translation)) == 0


def test_an_empty_room_has_no_obstacles(
    make_room: Room, make_pose: Pose
) -> None:
    depth, floor = make_room(H, 0.1, 8.0)  # the wall is beyond MAX_RANGE
    seen = _run(depth, floor, [], *make_pose(H, 0.1))
    assert seen.metric.from_floor
    assert len(seen.points) == 0


def test_a_wall_ahead_is_found_at_its_distance_from_relative_depth(
    make_room: Room, make_pose: Pose
) -> None:
    depth, floor = make_room(H, 0.1, 3.0)
    seen = _run(depth, floor, [], *make_pose(H, 0.1))
    assert seen.metric.from_floor
    assert len(seen.points) > 0
    np.testing.assert_allclose(seen.points[:, 0], 3.0, rtol=0.02)


def test_a_person_box_adds_its_points_to_the_cloud(
    make_room: Room, make_pose: Pose
) -> None:
    depth, floor = make_room(H, 0.1, 8.0)
    rotation, translation = make_pose(H, 0.1)
    box = _box("person", bottom=200.0)
    seen = _run(depth, floor, [box], rotation, translation)
    expected = pipeline.box_points([box], K, rotation, translation)
    assert seen.boxes == [box]
    assert len(seen.points) == len(expected) > 0
