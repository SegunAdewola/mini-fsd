"""Tests for minifsd.train.dino: a frozen DINOv2 with a floor head (E2b).

A tiny random DINOv2 on generated frames, on CPU. Needs torch:

    pixi run -e cv pytest tests/train/test_dino.py
"""

import json
import pathlib

import cv2
import numpy as np
import pytest

from minifsd.data import boxes

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from minifsd.perception import floor_dino  # noqa: E402  (needs torch)
from minifsd.train import dino, floor  # noqa: E402

N_FRAMES = 12


def _split(folder: pathlib.Path, seed: int) -> pathlib.Path:
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    images = []
    for i in range(N_FRAMES):
        stem = f"plan{seed:07d}_{i:04d}"
        rgb = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        mask = np.zeros((240, 320), np.uint8)
        mask[int(rng.integers(100, 200)) :] = 255
        cv2.imwrite(str(folder / f"{stem}_rgb.png"), rgb)
        cv2.imwrite(str(folder / f"{stem}_floor.png"), mask)
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


def _config(tmp_path: pathlib.Path, epochs: int) -> dino.Config:
    if not (tmp_path / "train").exists():
        _split(tmp_path / "train", 0)
        _split(tmp_path / "val", 1)
    return dino.Config(
        train=tmp_path / "train",
        val=tmp_path / "val",
        run_dir=tmp_path / "run",
        pretrained=None,  # tiny random backbone: no download
        epochs=epochs,
        batch_size=4,
        workers=0,
        device="cpu",
    )


def test_input_sides_are_patch_multiples_near_the_camera_size() -> None:
    x = floor_dino.preprocess(np.zeros((240, 320, 3), np.uint8))
    _, _, h, w = x.shape
    assert h % floor_dino.PATCH == 0 and w % floor_dino.PATCH == 0
    assert abs(h - 240) < floor_dino.PATCH and abs(w - 320) < floor_dino.PATCH


def test_the_head_gives_two_class_logits_on_the_patch_grid() -> None:
    model = dino.build_model(None)
    x = torch.from_numpy(
        floor_dino.preprocess(np.zeros((240, 320, 3), np.uint8))
    )
    logits = model(x)
    assert logits.shape == (
        1,
        2,
        floor_dino.INPUT_HEIGHT // floor_dino.PATCH,
        floor_dino.INPUT_WIDTH // floor_dino.PATCH,
    )


def test_training_moves_the_head_and_never_the_backbone(
    tmp_path: pathlib.Path,
) -> None:
    # The point of E2b: are frozen foundation features enough? So the
    # backbone must come out exactly as it went in.
    torch.manual_seed(0)
    before = {
        k: v.clone() for k, v in dino.build_model(None).state_dict().items()
    }
    config = _config(tmp_path, epochs=1)
    floor.train(config, dino.RECIPE)
    after = torch.load(config.run_dir / floor.LAST, weights_only=False)["model"]
    for key, value in before.items():
        if key.startswith("backbone."):
            assert torch.equal(value, after[key]), key
    assert any(
        not torch.equal(before[k], after[k])
        for k in before
        if k.startswith("classifier.")
    )


def test_only_the_head_is_given_to_the_optimiser() -> None:
    model = dino.build_model(None)
    groups = dino.RECIPE.param_groups(
        model,
        dino.Config(
            train=pathlib.Path(), val=pathlib.Path(), run_dir=pathlib.Path()
        ),
    )
    trained = {id(p) for g in groups for p in g["params"]}
    assert trained.isdisjoint(id(p) for p in model.backbone.parameters())
    assert trained == {
        id(p)
        for n, p in model.named_parameters()
        if not n.startswith("backbone.")
    }


def test_best_loads_back_as_a_floor_masker_and_resume_works(
    tmp_path: pathlib.Path,
) -> None:
    floor.train(_config(tmp_path, epochs=1), dino.RECIPE)
    floor.train(_config(tmp_path, epochs=2), dino.RECIPE)
    log = (tmp_path / "run" / floor.LOG).read_text().splitlines()
    assert [json.loads(line)["epoch"] for line in log] == [0, 1]

    model = floor_dino.FloorDino(tmp_path / "run" / floor.BEST)
    mask = model.predict(np.zeros((300, 401, 3), np.uint8))
    assert mask.shape == (300, 401) and mask.dtype == bool


def test_a_reloaded_model_gives_the_same_logits(tmp_path: pathlib.Path) -> None:
    torch.manual_seed(0)
    model = dino.build_model(None).eval()
    model.save(tmp_path / "m")
    again = floor_dino.DinoFloor.load(tmp_path / "m").eval()
    x = torch.rand(1, 3, floor_dino.INPUT_HEIGHT, floor_dino.INPUT_WIDTH)
    with torch.inference_mode():
        torch.testing.assert_close(model(x), again(x))
