"""Tests for minifsd.train.segformer: floor training that survives (E2a, T2).

A tiny random SegFormer on 20 generated frames, on CPU, so these run in
seconds and never download weights. They need torch:

    pixi run -e cv pytest tests/train/test_segformer.py
"""

from __future__ import annotations  # names below exist only with torch

import importlib.util
import json
import pathlib

import cv2
import numpy as np
import pytest

from minifsd.data import boxes

# A skip mark, not a module-level importorskip: ROS's launch_testing plugin
# imports every test file and lets a module-level skip (a BaseException)
# abort collection of the whole folder in the default env.
HAS_TORCH = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers")
)
pytestmark = pytest.mark.skipif(
    not HAS_TORCH, reason="needs torch and transformers (the cv env)"
)
if HAS_TORCH:
    import torch

    from minifsd.train import segformer

N_FRAMES = 20


def _split(folder: pathlib.Path, seed: int) -> pathlib.Path:
    """A v9-shaped split: rgb and floor PNGs plus coco.json."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images = []
    for i in range(N_FRAMES):
        stem = f"plan{seed:07d}_{i:04d}"
        rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        floor = np.zeros((240, 320), np.uint8)
        floor[int(rng.integers(100, 200)) :] = 255
        rgb[floor > 0] = (90, 80, 70)  # floor is a flat colour: learnable
        cv2.imwrite(str(folder / f"{stem}_rgb.png"), rgb[..., ::-1])
        cv2.imwrite(str(folder / f"{stem}_floor.png"), floor)
        images.append(
            {
                "id": i + 1,
                "file_name": f"{stem}_rgb.png",
                "width": 320,
                "height": 240,
            }
        )
    categories = [
        {"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)
    ]
    (folder / "coco.json").write_text(
        json.dumps(
            {"images": images, "annotations": [], "categories": categories}
        )
    )
    return folder


def _config(tmp_path: pathlib.Path, epochs: int) -> segformer.Config:
    train = tmp_path / "v9" / "train"
    val = tmp_path / "v9" / "val"
    if not train.exists():
        _split(train, 0)
        _split(val, 1)
    return segformer.Config(
        train=train,
        val=val,
        run_dir=tmp_path / "run",
        pretrained=None,  # tiny random model: no download
        epochs=epochs,
        batch_size=4,
        workers=0,
        device="cpu",
    )


def _log(run_dir: pathlib.Path) -> list[dict[str, float]]:
    lines = (run_dir / segformer.LOG).read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_an_epoch_writes_a_checkpoint_a_best_model_and_a_log_line(
    tmp_path: pathlib.Path,
) -> None:
    config = _config(tmp_path, epochs=1)
    segformer.train(config)
    assert (config.run_dir / segformer.LAST).exists()
    assert (config.run_dir / segformer.BEST / "config.json").exists()
    (row,) = _log(config.run_dir)
    assert row["epoch"] == 0
    assert 0.0 <= row["val_iou"] <= 1.0
    assert row["train_loss"] > 0
    assert row["device"] == "cpu"


def test_a_second_call_resumes_instead_of_starting_over(
    tmp_path: pathlib.Path,
) -> None:
    segformer.train(_config(tmp_path, epochs=1))
    first = torch.load(tmp_path / "run" / segformer.LAST, weights_only=False)[
        "model"
    ]
    segformer.train(_config(tmp_path, epochs=2))  # the session "dropped"
    assert [row["epoch"] for row in _log(tmp_path / "run")] == [0, 1]
    second = torch.load(tmp_path / "run" / segformer.LAST, weights_only=False)[
        "model"
    ]
    # Epoch 1 trained on from epoch 0's weights, so they moved on.
    assert any(not torch.equal(first[k], second[k]) for k in first)


def test_a_finished_run_does_nothing_when_called_again(
    tmp_path: pathlib.Path,
) -> None:
    segformer.train(_config(tmp_path, epochs=1))
    segformer.train(_config(tmp_path, epochs=1))
    assert len(_log(tmp_path / "run")) == 1


def test_training_lowers_the_loss_on_an_easy_task(
    tmp_path: pathlib.Path,
) -> None:
    # Floor is one flat colour here, so a working loop must learn something.
    config = _config(tmp_path, epochs=4)
    segformer.train(config)
    losses = [row["train_loss"] for row in _log(config.run_dir)]
    assert losses[-1] < losses[0]


def test_the_dataset_gives_normalised_images_and_0_1_labels(
    tmp_path: pathlib.Path,
) -> None:
    config = _config(tmp_path, epochs=1)
    data = segformer.FloorDataset(config.train, augment=False)
    assert len(data) == N_FRAMES
    x, y = data[0]
    assert x.shape == (3, 240, 320) and x.dtype == torch.float32
    assert y.shape == (240, 320) and y.dtype == torch.int64
    assert set(torch.unique(y).tolist()) <= {0, 1}


def test_limit_trains_on_the_first_n_frames(tmp_path: pathlib.Path) -> None:
    config = _config(tmp_path, epochs=1)
    data = segformer.FloorDataset(config.train, augment=False, limit=5)
    assert len(data) == 5
