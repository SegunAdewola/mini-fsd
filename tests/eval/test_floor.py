"""Tests for minifsd.eval.floor, with fake samples and fake models."""

import dataclasses
import math

import numpy as np
import pytest

from minifsd.eval import floor


@dataclasses.dataclass
class FakeSample:
    index: int
    rgb: np.ndarray
    floor: np.ndarray
    labeled: np.ndarray


def _sample(index: int, floor_rows: int) -> FakeSample:
    """A 10x10 image whose bottom floor_rows rows are floor.

    The bottom-centre check looks at row 4 (5 rows above the bottom), so
    floor_rows of 6 or more counts as "bottom is floor".
    """
    gt = np.zeros((10, 10), bool)
    if floor_rows:
        gt[-floor_rows:] = True
    return FakeSample(
        index, np.zeros((10, 10, 3), np.uint8), gt, np.ones((10, 10), bool)
    )


SAMPLES = [_sample(7, 6), _sample(3, 0), _sample(9, 8)]


def test_perfect_model_scores_one() -> None:
    truth = {id(s.rgb): s.floor for s in SAMPLES}
    result = floor.evaluate_floor(SAMPLES, lambda rgb: truth[id(rgb)])
    assert result.iou == 1.0
    assert result.n_images == 3


def test_model_that_never_finds_floor_scores_zero() -> None:
    result = floor.evaluate_floor(SAMPLES, lambda rgb: np.zeros((10, 10), bool))
    assert result.iou == 0.0


def test_iou_is_micro_averaged() -> None:
    # Predicts the bottom 4 rows everywhere.
    # Image 7: I=40, U=60. Image 3: I=0, U=40. Image 9: I=40, U=80.
    # Micro: 80 / 180 = 0.444. The mean of per-image IoUs would be
    # (0.667 + 0 + 0.5) / 3 = 0.389, a different number.
    pred = np.zeros((10, 10), bool)
    pred[-4:] = True
    result = floor.evaluate_floor(SAMPLES, lambda rgb: pred)
    assert result.iou == pytest.approx(80 / 180)
    assert dict(result.per_image) == pytest.approx({7: 40 / 60, 3: 0, 9: 0.5})


def test_bottom_floor_subset_skips_images_without_floor_there() -> None:
    pred = np.zeros((10, 10), bool)
    pred[-4:] = True
    result = floor.evaluate_floor(SAMPLES, lambda rgb: pred)
    assert result.n_bottom_floor == 2  # images 7 and 9
    assert result.iou_bottom_floor == pytest.approx(80 / 140)


def test_bottom_floor_iou_is_nan_when_no_image_qualifies() -> None:
    result = floor.evaluate_floor(
        [_sample(1, 0)], lambda rgb: np.zeros((10, 10), bool)
    )
    assert result.n_bottom_floor == 0
    assert math.isnan(result.iou_bottom_floor)


def test_worst_lists_lowest_iou_first() -> None:
    pred = np.zeros((10, 10), bool)
    pred[-4:] = True
    result = floor.evaluate_floor(SAMPLES, lambda rgb: pred)
    assert result.worst(2) == [3, 9]


def test_latency_is_recorded() -> None:
    result = floor.evaluate_floor(SAMPLES, lambda rgb: np.zeros((10, 10), bool))
    assert 0 <= result.p50_ms <= result.p95_ms


def test_no_samples_raises() -> None:
    with pytest.raises(ValueError):
        floor.evaluate_floor([], lambda rgb: rgb)
