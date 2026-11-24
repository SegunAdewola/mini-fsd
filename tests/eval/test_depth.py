"""Tests for minifsd.eval.depth: scoring a relative depth model."""

import dataclasses

import numpy as np
import pytest

from minifsd.eval import depth, depth_align


@dataclasses.dataclass(frozen=True)
class FakeSample:
    index: int
    rgb: np.ndarray
    depth: np.ndarray


def _samples() -> list[FakeSample]:
    rng = np.random.default_rng(1)
    return [
        FakeSample(i, rng.integers(0, 255, (6, 8, 3), np.uint8), gt)
        for i, gt in enumerate(rng.uniform(1, 8, (3, 6, 8)))
    ]


TRUE = depth_align.Fit(scale=2.0, shift=0.05)
BY_INDEX = {s.index: s for s in _samples()}


def _perfect(sample_rgb: np.ndarray) -> np.ndarray:
    # A perfect relative model: the right disparity up to TRUE's scale/shift.
    (s,) = [s for s in BY_INDEX.values() if s.rgb is sample_rgb]
    return (1.0 / s.depth - TRUE.shift) / TRUE.scale


def test_perfect_model_scores_perfectly_both_ways() -> None:
    result = depth.evaluate_depth(list(BY_INDEX.values()), _perfect, TRUE)
    assert result.n_images == 3
    assert result.per_image_fit.abs_rel == pytest.approx(0.0, abs=1e-5)
    assert result.per_image_fit.delta1 == pytest.approx(1.0)
    assert result.global_fit.abs_rel == pytest.approx(0.0, abs=1e-5)


def test_a_wrong_global_fit_shows_only_in_the_global_scores() -> None:
    wrong = depth_align.Fit(scale=TRUE.scale * 1.5, shift=TRUE.shift)
    result = depth.evaluate_depth(list(BY_INDEX.values()), _perfect, wrong)
    assert result.per_image_fit.abs_rel == pytest.approx(0.0, abs=1e-5)
    assert result.global_fit.abs_rel > 0.1


def test_reports_latency_and_per_image_scores() -> None:
    result = depth.evaluate_depth(list(BY_INDEX.values()), _perfect, TRUE)
    assert result.p50_ms >= 0 and result.p95_ms >= result.p50_ms
    assert [i for i, _ in result.per_image] == [0, 1, 2]


def test_rejects_no_samples() -> None:
    with pytest.raises(ValueError):
        depth.evaluate_depth([], _perfect, TRUE)
