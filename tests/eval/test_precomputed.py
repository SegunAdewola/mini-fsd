"""Tests for minifsd.eval.precomputed: models run elsewhere (E3a)."""

import dataclasses
import math
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.eval import harness, precomputed


@dataclasses.dataclass(frozen=True)
class Sample:
    index: int
    rgb: np.ndarray
    depth: np.ndarray


def _samples(n: int, seed: int = 0) -> list[Sample]:
    rng = np.random.default_rng(seed)
    return [
        Sample(
            i,
            rng.integers(0, 256, (12, 16, 3), dtype=np.uint8),
            rng.uniform(1, 5, (12, 16)).astype(np.float32),
        )
        for i in range(n)
    ]


@settings(deadline=None, max_examples=30)
@given(st.integers(0, 2**32 - 1))
def test_the_key_follows_the_pixels(seed: int) -> None:
    a = _samples(1, seed)[0].rgb
    b = a.copy()
    assert precomputed.key(a) == precomputed.key(b)
    b[0, 0, 0] ^= 1
    assert precomputed.key(a) != precomputed.key(b)
    # Same bytes, other shape: a different image.
    assert precomputed.key(a) != precomputed.key(a.reshape(16, 12, 3))


def test_export_writes_each_image_once_losslessly(
    tmp_path: pathlib.Path,
) -> None:
    samples = _samples(3)
    # A dataset seen twice (nyuv2-fit is inside nyuv2-train) adds nothing.
    n = precomputed.export_images([*samples, samples[0]], tmp_path)
    assert n == 3
    for s in samples:
        path = tmp_path / f"{precomputed.key(s.rgb)}.png"
        back = precomputed.read_rgb(path)
        np.testing.assert_array_equal(back, s.rgb)


def test_predictions_come_back_as_disparity_by_key(
    tmp_path: pathlib.Path,
) -> None:
    samples = _samples(2)
    depth = {precomputed.key(s.rgb): s.depth for s in samples}
    precomputed.save(tmp_path / "pred.npz", depth)
    model = precomputed.PrecomputedDepth(tmp_path / "pred.npz")
    for s in samples:
        np.testing.assert_allclose(
            model.predict(s.rgb), 1.0 / s.depth, rtol=1e-3
        )


def test_a_photo_with_no_prediction_is_an_error(tmp_path: pathlib.Path) -> None:
    precomputed.save(tmp_path / "pred.npz", {"x": np.ones((2, 2))})
    model = precomputed.PrecomputedDepth(tmp_path / "pred.npz")
    with pytest.raises(KeyError, match="no prediction"):
        model.predict(_samples(1)[0].rgb)


def test_an_untimed_candidate_logs_no_latency(tmp_path: pathlib.Path) -> None:
    # DA3 runs on Colab, so a lookup time would be a lie: none is logged.
    samples = _samples(4)
    precomputed.save(
        tmp_path / "pred.npz",
        {precomputed.key(s.rgb): s.depth for s in samples},
    )
    model = precomputed.PrecomputedDepth(tmp_path / "pred.npz")
    c = harness.Candidate("colab-depth", "depth", model.predict, timed=False)
    r = harness.evaluate(c, samples[2:], "fake", 2, fit=samples[:2])
    assert math.isnan(r.p50_ms) and math.isnan(r.p95_ms)
    assert r.metrics["abs_rel_oracle"] == pytest.approx(0.0, abs=1e-3)
