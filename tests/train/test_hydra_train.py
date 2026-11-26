"""Tests for minifsd.train.hydra: HydraNet-lite joint training (H4, E4).

A tiny random backbone on generated data, on CPU. Needs torch:

    pixi run -e cv pytest tests/train/test_hydra.py
"""

from __future__ import annotations  # names below exist only with torch

import importlib.util
import json
import pathlib

import cv2
import numpy as np
import pytest

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

    from minifsd.perception import centernet, floor_dino
    from minifsd.train import floor, hydra

DOOR = boxes.CATEGORIES.index("door") + 1  # COCO ids from 1
PERSON = boxes.CATEGORIES.index("person") + 1


def _coco(
    images: list[dict[str, object]], anns: list[dict[str, object]]
) -> str:
    categories = [
        {"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)
    ]
    return json.dumps(
        {"images": images, "annotations": anns, "categories": categories}
    )


def _sim(folder: pathlib.Path, n: int, seed: int) -> pathlib.Path:
    """A v10-shaped split: rgb, floor and depth PNGs, meta, one door each."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images, anns = [], []
    for i in range(n):
        stem = f"plan{seed:07d}_{i:04d}"
        rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        mask = np.zeros((240, 320), np.uint8)
        mask[160:] = 255
        rgb[mask > 0] = (90, 80, 70)
        depth_mm = np.full((240, 320), 3000, np.uint16)
        depth_mm[160:] = 1500
        cv2.imwrite(str(folder / f"{stem}_rgb.png"), rgb[..., ::-1])
        cv2.imwrite(str(folder / f"{stem}_floor.png"), mask)
        cv2.imwrite(str(folder / f"{stem}_depth.png"), depth_mm)
        images.append(
            {
                "id": i + 1,
                "file_name": f"{stem}_rgb.png",
                "width": 320,
                "height": 240,
            }
        )
        anns.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": DOOR,
                "bbox": [40.0, 40.0, 60.0, 120.0],
                "area": 7200,
                "iscrowd": 0,
            }
        )
    (folder / "coco.json").write_text(_coco(images, anns))
    (folder / "meta.jsonl").write_text("{}\n")
    return folder


def _people(folder: pathlib.Path, n: int) -> pathlib.Path:
    """A COCO-shaped folder: 640 x 480 JPEGs with one person each."""
    folder.mkdir(parents=True)
    rng = np.random.default_rng(9)
    images, anns = [], []
    for i in range(n):
        name = f"{i:012d}.jpg"
        cv2.imwrite(
            str(folder / name), rng.integers(0, 256, (480, 640, 3), np.uint8)
        )
        images.append(
            {"id": i + 1, "file_name": name, "width": 640, "height": 480}
        )
        anns.append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": PERSON,
                "bbox": [320.0, 120.0, 80.0, 240.0],
                "area": 19200,
                "iscrowd": 0,
            }
        )
    (folder / "coco.json").write_text(_coco(images, anns))
    return folder


def _config(tmp_path: pathlib.Path, epochs: int) -> hydra.Config:
    if not (tmp_path / "v10").exists():
        _sim(tmp_path / "v10" / "train", 6, 0)
        _sim(tmp_path / "v10" / "val", 4, 1)
        _people(tmp_path / "coco" / "train", 4)
    return hydra.Config(
        train=[tmp_path / "v10" / "train", tmp_path / "coco" / "train"],
        val=[tmp_path / "v10" / "val"],
        run_dir=tmp_path / "run",
        pretrained=None,
        epochs=epochs,
        batch_size=4,
        workers=0,
        device="cpu",
    )


def _peak(heatmap: torch.Tensor) -> tuple[int, int]:
    row, col = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    return int(row), int(col)


def test_a_sim_frame_labels_every_task_and_class(
    tmp_path: pathlib.Path,
) -> None:
    data = hydra.HydraDataset([_sim(tmp_path / "s", 1, 0)], augment=False)
    item = data[0]
    assert item["x"].shape == (
        3,
        floor_dino.INPUT_HEIGHT,
        floor_dino.INPUT_WIDTH,
    )
    assert bool(item["has_floor"]) and bool(item["has_depth"])
    assert item["class_mask"].all()
    assert set(torch.unique(item["floor"]).tolist()) == {0, 1}
    assert float(item["depth"][200, 10]) == pytest.approx(1.5)
    # The door's centre (70, 100) in a 320 x 240 frame, scaled to 322 x 238.
    sx, sy = floor_dino.INPUT_WIDTH / 320, floor_dino.INPUT_HEIGHT / 240
    expected = (
        int(100 * sy / centernet.STRIDE),
        int(70 * sx / centernet.STRIDE),
    )
    assert _peak(item["heatmap"][DOOR - 1]) == expected


def test_a_coco_photo_labels_boxes_only_and_not_doors_or_signs(
    tmp_path: pathlib.Path,
) -> None:
    item = hydra.HydraDataset([_people(tmp_path / "c", 1)], augment=False)[0]
    assert not bool(item["has_floor"]) and not bool(item["has_depth"])
    mask = dict(zip(boxes.CATEGORIES, item["class_mask"].tolist(), strict=True))
    assert mask == {
        "obstacle": True,
        "door": False,
        "room_sign": False,
        "person": True,
    }
    assert float(item["heatmap"][PERSON - 1].max()) == 1.0


def test_a_flip_moves_the_box_with_the_image(tmp_path: pathlib.Path) -> None:
    folder = _sim(tmp_path / "s", 1, 0)
    plain = hydra.HydraDataset([folder], augment=False)[0]
    flipped = hydra.flip(plain)
    torch.testing.assert_close(flipped["x"], plain["x"].flip(-1))
    torch.testing.assert_close(flipped["floor"], plain["floor"].flip(-1))
    row, col = _peak(plain["heatmap"][DOOR - 1])
    cols = plain["heatmap"].shape[-1]
    new_row, new_col = _peak(flipped["heatmap"][DOOR - 1])
    assert new_row == row and abs(new_col - (cols - 1 - col)) <= 1


def test_training_writes_checkpoints_and_resumes(
    tmp_path: pathlib.Path,
) -> None:
    hydra.train(_config(tmp_path, epochs=1))
    run = tmp_path / "run"
    assert (run / floor.LAST).exists()
    assert (run / floor.BEST / "heads.pt").exists()
    hydra.train(_config(tmp_path, epochs=2))  # the session "dropped"
    log = [json.loads(x) for x in (run / floor.LOG).read_text().splitlines()]
    assert [row["epoch"] for row in log] == [0, 1]
    assert all(row["val_loss"] > 0 for row in log)


def test_the_loss_falls_on_an_easy_task(tmp_path: pathlib.Path) -> None:
    config = _config(tmp_path, epochs=4)
    hydra.train(config)
    log = [
        json.loads(x)
        for x in (config.run_dir / floor.LOG).read_text().splitlines()
    ]
    assert log[-1]["train_loss"] < log[0]["train_loss"]
