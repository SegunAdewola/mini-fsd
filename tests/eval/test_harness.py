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


def test_the_segformer_candidate_loads_a_checkpoint_and_masks_floor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # E2a: a run's best/ folder, copied to MODELS/<name>, scored as "floor".
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.train import segformer

    segformer.build_model(None).save_pretrained(tmp_path / "floor-segformer-b0")
    monkeypatch.setattr(harness, "MODELS", tmp_path)
    c = harness.make_candidate("floor-segformer-b0", device="cpu", threads=1)
    assert (c.name, c.task) == ("floor-segformer-b0", "floor")
    mask = c.predict(np.zeros((240, 320, 3), np.uint8))
    assert mask.shape == (240, 320) and mask.dtype == bool


# NYUv2's official split: 795 train and 654 test images, 1-based indices into
# the 1,449 labeled ones. Stand-ins with the same sizes, so the index choice
# is tested without the 3 GB file.
NYU_SPLITS = {
    "train": list(range(1, 796)),
    "test": list(range(796, 1450)),
}


def test_nyuv2_val_comes_from_train_and_never_from_the_fit_sample() -> None:
    val = harness.nyuv2_indices("nyuv2-val", NYU_SPLITS)
    fit = harness.nyuv2_indices("nyuv2-fit", NYU_SPLITS)
    assert len(val) == harness.VAL_IMAGES
    assert len(set(val)) == len(val)
    assert set(val) <= set(NYU_SPLITS["train"])
    assert not set(val) & set(fit)
    assert not set(val) & set(NYU_SPLITS["test"])
    assert val == sorted(val)
    assert val == harness.nyuv2_indices("nyuv2-val", NYU_SPLITS)


def test_the_fit_sample_is_still_the_one_v5_used() -> None:
    # V5's depth numbers were fitted on this sample; adding nyuv2-val must
    # not move it.
    rng = np.random.default_rng(0)
    v5 = sorted(
        int(i)
        for i in rng.choice(
            NYU_SPLITS["train"], harness.FIT_IMAGES, replace=False
        )
    )
    assert harness.nyuv2_indices("nyuv2-fit", NYU_SPLITS) == v5


def test_nyuv2_train_and_test_are_the_official_splits() -> None:
    for split in ("train", "test"):
        assert (
            harness.nyuv2_indices(f"nyuv2-{split}", NYU_SPLITS)
            == NYU_SPLITS[split]
        )
    with pytest.raises(ValueError):
        harness.nyuv2_indices("nyuv2-other", NYU_SPLITS)


def test_the_dino_candidate_loads_a_checkpoint_and_masks_floor(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # E2b: a run's best/ folder, copied to MODELS/<name>, scored as "floor".
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.train import dino

    dino.build_model(None).save(tmp_path / "floor-dinov2-s")
    monkeypatch.setattr(harness, "MODELS", tmp_path)
    c = harness.make_candidate("floor-dinov2-s", device="cpu", threads=1)
    assert (c.name, c.task) == ("floor-dinov2-s", "floor")
    mask = c.predict(np.zeros((240, 320, 3), np.uint8))
    assert mask.shape == (240, 320) and mask.dtype == bool


def test_the_yolo_candidate_runs_the_exported_model_under_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # E1a: the ONNX export, run by ONNX Runtime with the harness's threads.
    from minifsd.perception import detect_onnx

    opened: list[tuple[pathlib.Path, int]] = []

    class Fake:
        def __init__(self, path: pathlib.Path, threads: int) -> None:
            opened.append((path, threads))

        def predict(self, rgb: np.ndarray) -> list[object]:
            return []

    monkeypatch.setattr(detect_onnx, "OnnxDetector", Fake)
    c = harness.make_candidate("detect-yolo26n", device="cpu", threads=2)
    assert (c.name, c.task) == ("detect-yolo26n", "detect")
    assert opened == [(harness.YOLO_ONNX, 2)]
    assert c.predict(np.zeros((240, 320, 3), np.uint8)) == []


@pytest.mark.parametrize(
    ("name", "kind"),
    [("floor-dinov2-s-v10", "dino"), ("floor-segformer-b0-v10", "segformer")],
)
def test_v10_retrains_load_from_their_own_folders(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    kind: str,
) -> None:
    # E2c: each model loads from MODELS/<its name>, so v9 and v10 runs of
    # one architecture are scored side by side.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.train import dino, segformer

    if kind == "dino":
        dino.build_model(None).save(tmp_path / name)
    else:
        segformer.build_model(None).save_pretrained(tmp_path / name)
    monkeypatch.setattr(harness, "MODELS", tmp_path)
    c = harness.make_candidate(name, device="cpu", threads=1)
    assert (c.name, c.task) == (name, "floor")
    assert c.predict(np.zeros((240, 320, 3), np.uint8)).shape == (240, 320)


def test_the_rf_detr_candidate_decodes_as_detr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # E1b: same runtime and thread cap as YOLO, its own output layout.
    from minifsd.perception import detect_onnx

    opened: list[tuple[pathlib.Path, int, str]] = []

    class Fake:
        def __init__(
            self, path: pathlib.Path, threads: int, kind: str = "yolo"
        ) -> None:
            opened.append((path, threads, kind))

        def predict(self, rgb: np.ndarray) -> list[object]:
            return []

    monkeypatch.setattr(detect_onnx, "OnnxDetector", Fake)
    c = harness.make_candidate("detect-rfdetr-nano", device="cpu", threads=2)
    assert (c.name, c.task) == ("detect-rfdetr-nano", "detect")
    assert opened == [(harness.RFDETR_ONNX, 2, "detr")]


def test_the_da3_candidate_reads_colab_predictions_untimed(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # E3a: DA3 ran on Colab; the harness only looks its depth maps up.
    from minifsd.eval import precomputed

    rgb = np.zeros((12, 16, 3), np.uint8)
    precomputed.save(
        tmp_path / "da3.npz", {precomputed.key(rgb): np.full((12, 16), 2.0)}
    )
    monkeypatch.setattr(harness, "DA3_PREDICTIONS", tmp_path / "da3.npz")
    c = harness.make_candidate("depth-da3-small", device="cpu", threads=2)
    assert (c.name, c.task, c.timed) == ("depth-da3-small", "depth", False)
    np.testing.assert_allclose(c.predict(rgb), 0.5, rtol=1e-3)


@pytest.mark.parametrize(
    ("name", "task"),
    [
        ("hydra-floor", "floor"),
        ("hydra-detect", "detect"),
        ("hydra-depth", "depth"),
        ("hydra-all", "floor"),
    ],
)
def test_hydra_candidates_answer_their_task_from_one_model(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    task: str,
) -> None:
    # E4: one trained HydraNet scored per task; hydra-all times every head.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.train import hydra as hydra_train

    hydra_train.build_model(None).save(tmp_path / "hydranet")
    monkeypatch.setattr(harness, "MODELS", tmp_path)
    c = harness.make_candidate(name, device="cpu", threads=1)
    assert (c.name, c.task) == (name, task)
    out = c.predict(np.zeros((240, 320, 3), np.uint8))
    if task == "floor":
        assert out.shape == (240, 320) and out.dtype == bool
    elif task == "depth":
        assert out.shape == (240, 320) and (out > 0).all()
    else:
        assert isinstance(out, list)


def test_task_candidates_skip_the_heads_they_dont_need(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # hydra-floor is timed as floor alone; hydra-all pays for every head.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.perception import hydra
    from minifsd.train import hydra as hydra_train

    hydra_train.build_model(None).save(tmp_path / "hydranet")
    monkeypatch.setattr(harness, "MODELS", tmp_path)
    used: list[tuple[str, ...]] = []
    real = hydra.Hydra.predict

    def spy(
        self: hydra.Hydra, rgb: np.ndarray, heads: tuple[str, ...]
    ) -> object:
        used.append(heads)
        return real(self, rgb, heads)

    monkeypatch.setattr(hydra.Hydra, "predict", spy)
    rgb = np.zeros((240, 320, 3), np.uint8)
    harness.make_candidate("hydra-floor", device="cpu", threads=1).predict(rgb)
    harness.make_candidate("hydra-all", device="cpu", threads=1).predict(rgb)
    assert used == [("floor",), ("floor", "depth", "detect")]


def test_the_teacher_set_is_train_without_val_or_fit() -> None:
    # E3b learns from these with the teacher's labels; nyuv2-val picks the
    # model and nyuv2-fit sets depth scale, so neither may be in it.
    teach = harness.nyuv2_indices("nyuv2-teach", NYU_SPLITS)
    val = harness.nyuv2_indices("nyuv2-val", NYU_SPLITS)
    fit = harness.nyuv2_indices("nyuv2-fit", NYU_SPLITS)
    assert set(teach) == set(NYU_SPLITS["train"]) - set(val) - set(fit)
    assert len(teach) == 795 - harness.VAL_IMAGES - harness.FIT_IMAGES
    assert teach == sorted(teach)


def test_the_depth_student_candidate_loads_its_folder(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # E3b: SegFormer-B0 at 160 x 120 predicting disparity, timed on the M1.
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.train import depth_student

    depth_student.build_model(None).save_pretrained(tmp_path / "depth-student")
    monkeypatch.setattr(harness, "MODELS", tmp_path)
    c = harness.make_candidate("depth-student", device="cpu", threads=1)
    assert (c.name, c.task, c.timed) == ("depth-student", "depth", True)
    assert c.predict(np.zeros((240, 320, 3), np.uint8)).shape == (240, 320)
