"""Tests for minifsd.eval.harness: one fair way to score every candidate."""

import dataclasses
import json
import os
import pathlib
import time

import cv2
import numpy as np
import pytest

from minifsd.eval import harness

SLEEP = 0.02  # seconds per image for the fake candidates


@dataclasses.dataclass(frozen=True)
class FakeSample:
    index: int
    rgb: np.ndarray
    depth: np.ndarray
    floor: np.ndarray
    labeled: np.ndarray


def _samples(n: int = 6) -> list[FakeSample]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        floor = np.zeros((12, 16), dtype=bool)
        floor[6 + i % 3 :] = True
        out.append(
            FakeSample(
                index=i,
                rgb=rng.integers(0, 255, (12, 16, 3), np.uint8),
                depth=rng.uniform(1, 8, (12, 16)),
                floor=floor,
                labeled=np.ones((12, 16), dtype=bool),
            )
        )
    return out


SAMPLES = _samples()
BY_RGB = {id(s.rgb): s for s in SAMPLES}


def _perfect_floor(rgb: np.ndarray) -> np.ndarray:
    time.sleep(SLEEP)
    return BY_RGB[id(rgb)].floor


def _perfect_depth(rgb: np.ndarray) -> np.ndarray:
    time.sleep(SLEEP)
    return 1.0 / BY_RGB[id(rgb)].depth  # disparity, scale 1, shift 0


def test_scores_a_perfect_floor_candidate() -> None:
    c = harness.Candidate("fake-floor", "floor", _perfect_floor)
    r = harness.evaluate(c, SAMPLES, dataset="fake", threads=2)
    assert r.n_images == len(SAMPLES)
    assert r.metrics["iou"] == pytest.approx(1.0)
    # A sleep never returns early, so SLEEP is a hard floor. The ceiling
    # only has to tell one image from the whole run (which would be
    # len(SAMPLES) x SLEEP); a tighter one failed on busy shared CI Macs,
    # where a 20 ms sleep took 51 ms.
    assert SLEEP * 1000 <= r.p50_ms < SLEEP * 1000 * len(SAMPLES)
    assert r.p95_ms >= r.p50_ms
    assert r.peak_rss_mb > 0
    assert (r.candidate, r.task, r.dataset, r.threads) == (
        "fake-floor",
        "floor",
        "fake",
        2,
    )


def test_scores_a_perfect_depth_candidate_both_ways() -> None:
    c = harness.Candidate("fake-depth", "depth", _perfect_depth)
    r = harness.evaluate(c, SAMPLES, dataset="fake", threads=1, fit=SAMPLES)
    assert r.metrics["abs_rel_oracle"] == pytest.approx(0.0, abs=1e-5)
    assert r.metrics["abs_rel_global"] == pytest.approx(0.0, abs=1e-5)
    assert r.metrics["delta1_global"] == pytest.approx(1.0)


def test_depth_needs_images_for_the_global_fit() -> None:
    c = harness.Candidate("fake-depth", "depth", _perfect_depth)
    with pytest.raises(ValueError):
        harness.evaluate(c, SAMPLES, dataset="fake", threads=1)


def test_thread_cap_reaches_the_libraries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in harness.THREAD_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    before = cv2.getNumThreads()
    try:
        harness.set_thread_cap(2)
        assert all(os.environ[v] == "2" for v in harness.THREAD_ENV_VARS)
        # The budget is "at most n". OpenCV on macOS (GCD backend) can only
        # run on 1 thread or on all of them, so the cap makes it 1.
        assert cv2.getNumThreads() <= 2
    finally:
        cv2.setNumThreads(before)


def test_results_are_logged_one_json_line_per_run(
    tmp_path: pathlib.Path,
) -> None:
    c = harness.Candidate("fake-floor", "floor", _perfect_floor)
    r = harness.evaluate(c, SAMPLES[:2], dataset="fake", threads=2)
    log = tmp_path / "results.jsonl"
    harness.append_result(r, log)
    harness.append_result(r, log)
    lines = log.read_text().splitlines()
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row["candidate"] == "fake-floor" and row["metrics"]["iou"] == 1.0
    assert {"git_commit", "time", "threads", "p50_ms", "peak_rss_mb"} <= set(
        row
    )


def test_unknown_names_are_rejected() -> None:
    with pytest.raises(ValueError):
        harness.make_candidate("no-such-model", device="cpu", threads=1)
    with pytest.raises(ValueError):
        list(harness.load_dataset("no-such-data"))


def test_input_size_is_only_for_models_that_have_one() -> None:
    with pytest.raises(ValueError):
        harness.make_candidate(
            "floor-classical", device="cpu", threads=1, input_size=266
        )


def test_input_size_goes_into_the_candidate_name() -> None:
    # The name is what the results log groups by, so each size of the
    # sweep must get its own.
    pytest.importorskip("torch")
    if not pathlib.Path(
        "data/models/depth-anything-v2-small/model.safetensors"
    ).is_file():
        pytest.skip("weights not downloaded")
    c = harness.make_candidate(
        "depth-anything-v2-small", device="cpu", threads=1, input_size=266
    )
    assert c.name == "depth-anything-v2-small@266"
    assert c.predict(np.zeros((240, 320, 3), np.uint8)).shape == (240, 320)


def _coco_folder(tmp_path: pathlib.Path) -> pathlib.Path:
    """Three tiny images and a COCO file, in the layout our data uses."""
    from minifsd.data import boxes

    found = [
        [boxes.Box2D("door", "d0", 2, 3, 10, 12, 100)],
        [
            boxes.Box2D("room_sign", "s0", 20, 4, 6, 4, 24),
            boxes.Box2D("obstacle", "c0", 1, 10, 8, 5, 40),
        ],
        [],
    ]
    images = []
    for i, b in enumerate(found):
        name = f"img{i}.png"
        cv2.imwrite(
            str(tmp_path / name), np.full((24, 32, 3), 40 * i, np.uint8)
        )
        images.append((name, 32, 24, b))
    (tmp_path / "coco.json").write_text(json.dumps(boxes.to_coco(images)))
    return tmp_path


def test_scores_a_perfect_detector_from_a_coco_folder(
    tmp_path: pathlib.Path,
) -> None:
    from minifsd.eval import detection

    folder = _coco_folder(tmp_path)
    samples = list(harness.load_dataset(f"coco:{folder}"))
    assert len(samples) == 3
    assert samples[0].rgb.shape == (24, 32, 3)
    # The fake detector looks its image up by mean colour.
    answers = {int(s.rgb.mean()): s.truth for s in samples}

    def perfect(rgb: np.ndarray) -> list[detection.Detection]:
        return [
            detection.Detection(t.category, t.x, t.y, t.w, t.h, 1.0)
            for t in answers[int(rgb.mean())]
        ]

    c = harness.Candidate("fake-detector", "detect", perfect)
    r = harness.evaluate(c, samples, dataset="fake", threads=1)
    assert r.task == "detect" and r.n_images == 3
    assert r.metrics["map50"] == pytest.approx(1.0)
    assert set(r.metrics) == {
        "map50",
        "ap50_door",
        "ap50_obstacle",
        "ap50_room_sign",
    }
    assert r.p95_ms >= r.p50_ms >= 0


def test_coco_folders_keep_crowd_regions(tmp_path: pathlib.Path) -> None:
    folder = _coco_folder(tmp_path)
    coco = json.loads((folder / "coco.json").read_text())
    coco["annotations"][0]["iscrowd"] = 1
    (folder / "coco.json").write_text(json.dumps(coco))
    first = next(iter(harness.load_dataset(f"coco:{folder}")))
    assert first.truth[0].crowd
