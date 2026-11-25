"""Tests for minifsd.eval.detection: mAP at IoU 0.5, VOC-style (T1)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.eval import detection as det

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


def _truth(seed: int, n_images: int = 6) -> list[list[det.Truth]]:
    """Random non-overlapping boxes, two categories, a few per image."""
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(n_images):
        boxes = []
        for k in range(int(rng.integers(1, 5))):
            category = ("door", "obstacle")[int(rng.integers(2))]
            x = 70.0 * k + float(rng.uniform(0, 10))
            boxes.append(det.Truth(category, x, 20.0, 50.0, 60.0))
        images.append(boxes)
    return images


def _perfect(truth: list[list[det.Truth]]) -> list[list[det.Detection]]:
    return [
        [det.Detection(t.category, t.x, t.y, t.w, t.h, 0.9) for t in image]
        for image in truth
    ]


def test_iou_of_boxes() -> None:
    a = det.Truth("door", 0, 0, 10, 10)
    assert det.iou(a, a) == pytest.approx(1.0)
    assert det.iou(a, det.Truth("door", 5, 0, 10, 10)) == pytest.approx(
        50 / 150
    )
    assert det.iou(a, det.Truth("door", 20, 0, 10, 10)) == 0.0


@settings(deadline=None)
@given(SEEDS)
def test_perfect_detections_score_one(seed: int) -> None:
    truth = _truth(seed)
    scores = det.mean_average_precision(_perfect(truth), truth)
    assert scores["map50"] == pytest.approx(1.0)
    for category in {t.category for image in truth for t in image}:
        assert scores[f"ap50_{category}"] == pytest.approx(1.0)


@settings(deadline=None)
@given(SEEDS)
def test_boxes_shifted_past_half_overlap_score_zero(seed: int) -> None:
    truth = _truth(seed)
    # 50 px wide boxes moved 30 px: IoU = 20 / 80 = 0.25.
    shifted = [
        [
            det.Detection(d.category, d.x + 30, d.y, d.w, d.h, d.score)
            for d in image
        ]
        for image in _perfect(truth)
    ]
    assert det.mean_average_precision(shifted, truth)["map50"] == 0.0


@settings(deadline=None)
@given(SEEDS, st.integers(min_value=0, max_value=3))
def test_a_missed_object_costs_its_share_of_recall(
    seed: int, drop: int
) -> None:
    truth = [[det.Truth("door", 70.0 * k, 0, 50, 60) for k in range(4)]]
    found = _perfect(truth)
    found[0] = found[0][drop:]  # miss `drop` of the 4 doors
    ap = det.mean_average_precision(found, truth)["ap50_door"]
    assert ap == pytest.approx((4 - drop) / 4)


@settings(deadline=None)
@given(SEEDS)
def test_false_alarms_below_every_hit_cost_nothing(seed: int) -> None:
    # All-point AP: past full recall, extra detections can't lower it.
    truth = _truth(seed)
    found = _perfect(truth)
    found[0] = [*found[0], det.Detection("door", 900, 900, 30, 30, 0.1)]
    assert det.mean_average_precision(found, truth)["map50"] == pytest.approx(
        1.0
    )


def test_a_confident_false_alarm_lowers_precision() -> None:
    truth = [[det.Truth("door", 0, 0, 50, 60)]]
    found = [
        [
            det.Detection("door", 500, 500, 50, 60, 0.99),  # wrong, confident
            det.Detection("door", 0, 0, 50, 60, 0.5),
        ]
    ]
    # Recall reaches 1 at the second detection, with precision 1/2.
    assert det.mean_average_precision(found, truth)[
        "ap50_door"
    ] == pytest.approx(0.5)


def test_a_second_detection_of_one_object_is_a_false_alarm() -> None:
    truth = [[det.Truth("door", 0, 0, 50, 60)]]
    found = [
        [
            det.Detection("door", 2, 0, 50, 60, 0.4),
            det.Detection("door", 0, 0, 50, 60, 0.9),
        ]
    ]
    # The better-scored one takes the match; AP is still 1, since the
    # duplicate comes after full recall. Reversed scores would hurt.
    assert det.mean_average_precision(found, truth)["ap50_door"] == 1.0
    found[0][0] = det.Detection("door", 2, 0, 50, 60, 0.95)
    found[0][1] = det.Detection("door", 400, 0, 50, 60, 0.9)  # now a miss
    assert det.mean_average_precision(found, truth)["ap50_door"] == 1.0


def test_detections_in_a_crowd_region_are_neither_hits_nor_alarms() -> None:
    # COCO's rule: a crowd region isn't a target, and matching it isn't a
    # mistake. Only the one real person counts.
    truth = [
        [
            det.Truth("person", 0, 0, 40, 80),
            det.Truth("person", 200, 0, 200, 100, crowd=True),
        ]
    ]
    found = [
        [
            det.Detection("person", 220, 10, 40, 80, 0.99),  # inside the crowd
            det.Detection("person", 0, 0, 40, 80, 0.8),
        ]
    ]
    assert det.mean_average_precision(found, truth)["ap50_person"] == 1.0


def test_a_category_with_no_objects_is_left_out_of_the_mean() -> None:
    truth = [[det.Truth("door", 0, 0, 50, 60)]]
    found = [[det.Detection("door", 0, 0, 50, 60, 0.9)]]
    found[0].append(det.Detection("room_sign", 100, 0, 10, 10, 0.9))
    scores = det.mean_average_precision(found, truth)
    assert scores["map50"] == 1.0
    assert "ap50_room_sign" not in scores
    assert not any(math.isnan(v) for v in scores.values())


def test_no_objects_at_all_is_an_error() -> None:
    with pytest.raises(ValueError):
        det.mean_average_precision([[]], [[]])
