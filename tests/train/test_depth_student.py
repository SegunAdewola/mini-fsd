"""Tests for minifsd.train.depth_student: E3b's distilled depth network.

A tiny random SegFormer, on CPU. Needs torch:

    pixi run -e cv pytest tests/train/test_depth_student.py
"""

from __future__ import annotations  # names below exist only with torch

import dataclasses
import importlib.util
import json
import pathlib

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.data import boxes

HAS_TORCH = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers")
)
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="needs torch and transformers (the cv env)"
)
if HAS_TORCH:
    import torch

    from minifsd.perception import depth_student as student
    from minifsd.train import depth_student as ds
    from minifsd.train import floor


@settings(deadline=None, max_examples=20)
@given(st.floats(0.05, 20.0), st.floats(-3.0, 3.0))
def test_the_loss_ignores_any_scale_and_shift(
    scale: float, shift: float
) -> None:
    gen = torch.Generator().manual_seed(0)
    target = torch.rand(2, 12, 16, generator=gen) + 0.1
    valid = torch.ones(2, 12, 16, dtype=torch.bool)
    pred = (target * scale + shift).unsqueeze(1)
    assert float(ds.ssi_loss(pred, target, valid)) == pytest.approx(
        0.0, abs=1e-4
    )


def test_the_loss_skips_invalid_pixels() -> None:
    target = torch.full((1, 4, 4), 0.5)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 0] = False
    target[0, 0, 0] = 2.0  # varies, so the alignment is well posed
    target[0, 1, 1] = 1.0
    pred = target.clone().unsqueeze(1)
    pred[0, 0, 0] = 99.0  # wrong, but on an invalid row
    pred[0, 0, 0, 0] = 99.0
    assert float(ds.ssi_loss(pred, target, valid)) == pytest.approx(
        0.0, abs=1e-4
    )


@dataclasses.dataclass(frozen=True)
class Sample:
    index: int
    rgb: np.ndarray


def test_the_teacher_export_writes_each_image_and_its_disparity(
    tmp_path: pathlib.Path,
) -> None:
    rng = np.random.default_rng(0)
    samples = [
        Sample(i, rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
        for i in (3, 7)
    ]
    n = ds.export_teacher(samples, lambda rgb: rgb[..., 0] / 255 + 1, tmp_path)
    assert n == 2
    item = ds.TeacherDepth(tmp_path)[1]
    expected = cv2.resize(
        (samples[1].rgb[..., 0] / 255 + 1).astype(np.float32),
        (ds.TARGET_WIDTH, ds.TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    np.testing.assert_allclose(item["target"].numpy(), expected, rtol=2e-3)
    assert item["valid"].all()


def _sim(folder: pathlib.Path, n: int, seed: int) -> pathlib.Path:
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images = []
    for i in range(n):
        stem = f"plan{seed:07d}_{i:04d}"
        rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        depth = np.full((240, 320), 3000, np.uint16)
        depth[120:] = 1500
        depth[0, 0] = 0  # no reading
        rgb[120:] = (90, 80, 70)
        cv2.imwrite(str(folder / f"{stem}_rgb.png"), rgb[..., ::-1])
        cv2.imwrite(str(folder / f"{stem}_depth.png"), depth)
        images.append(
            {
                "id": i + 1,
                "file_name": f"{stem}_rgb.png",
                "width": 320,
                "height": 240,
            }
        )
    categories = [
        {"id": i + 1, "name": c} for i, c in enumerate(boxes.CATEGORIES)
    ]
    (folder / "coco.json").write_text(
        json.dumps(
            {"images": images, "annotations": [], "categories": categories}
        )
    )
    return folder


def test_a_sim_frame_targets_one_over_depth_where_there_is_a_reading(
    tmp_path: pathlib.Path,
) -> None:
    item = ds.SimDepth(_sim(tmp_path / "s", 1, 0))[0]
    assert item["x"].shape == (3, student.INPUT_HEIGHT, student.INPUT_WIDTH)
    assert item["target"].shape == (ds.TARGET_HEIGHT, ds.TARGET_WIDTH)
    assert float(item["target"][200, 10]) == pytest.approx(1 / 1.5)
    assert not bool(item["valid"][0, 0]) and bool(item["valid"][100, 100])


def test_training_writes_checkpoints_resumes_and_the_candidate_predicts(
    tmp_path: pathlib.Path,
) -> None:
    sim = _sim(tmp_path / "train", 6, 0)
    rng = np.random.default_rng(1)
    teach = tmp_path / "teach"
    ds.export_teacher(
        [
            Sample(i, rng.integers(0, 256, (48, 64, 3), np.uint8))
            for i in range(3)
        ],
        lambda rgb: rgb[..., 1] / 255 + 1,
        teach,
    )
    config = ds.Config(
        sim=[sim],
        teacher=[teach],
        val=[_sim(tmp_path / "val", 3, 1)],
        run_dir=tmp_path / "run",
        pretrained=None,
        epochs=1,
        batch_size=4,
        workers=0,
        device="cpu",
    )
    ds.train(config)
    ds.train(dataclasses.replace(config, epochs=2))
    log = [
        json.loads(x)
        for x in (config.run_dir / floor.LOG).read_text().splitlines()
    ]
    assert [row["epoch"] for row in log] == [0, 1]
    model = student.DepthStudent(config.run_dir / floor.BEST)
    out = model.predict(np.zeros((300, 401, 3), np.uint8))
    assert out.shape == (300, 401) and out.dtype == np.float32
