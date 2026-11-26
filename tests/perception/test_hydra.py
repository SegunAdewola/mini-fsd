"""Tests for minifsd.perception.hydra: HydraNet-lite (H3, E4).

A tiny random DINOv2 backbone, on CPU. Needs torch:

    pixi run -e cv pytest tests/perception/test_hydra.py
"""

from __future__ import annotations  # names below exist only with torch

import importlib.util
import pathlib

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
    from transformers import Dinov2Config, Dinov2Model

    from minifsd.perception import centernet, floor_dino, hydra

N_CLASSES = len(boxes.CATEGORIES)


def _model() -> hydra.HydraNet:
    torch.manual_seed(0)
    backbone = Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=64,
            patch_size=floor_dino.PATCH,
            image_size=floor_dino.PATCH * 4,
        )
    )
    model = hydra.HydraNet(backbone, n_classes=N_CLASSES, neck=16)
    model.eval()
    return model


def _input() -> torch.Tensor:
    rgb = np.random.default_rng(0).integers(0, 256, (240, 320, 3), np.uint8)
    return torch.from_numpy(floor_dino.preprocess(rgb))


def test_one_forward_gives_every_head_on_its_grid() -> None:
    out = _model()(_input())
    patches = (
        floor_dino.INPUT_HEIGHT // floor_dino.PATCH,
        floor_dino.INPUT_WIDTH // floor_dino.PATCH,
    )
    dense = (
        floor_dino.INPUT_HEIGHT // centernet.STRIDE,
        floor_dino.INPUT_WIDTH // centernet.STRIDE,
    )
    assert out.floor.shape == (1, 2, *patches)
    assert out.log_depth.shape == (1, 1, *dense)
    assert out.heatmap.shape == (1, N_CLASSES, *dense)
    assert out.size.shape == (1, 2, *dense)
    assert out.offset.shape == (1, 2, *dense)


def test_heatmaps_start_near_ten_percent() -> None:
    # CenterNet's bias of -2.19: every cell starts at about 0.1, so the
    # focal loss doesn't explode on the first steps.
    p = torch.sigmoid(_model()(_input()).heatmap.detach())
    assert float(p.mean()) == pytest.approx(0.1, abs=0.03)


def test_the_backbone_is_frozen_and_the_heads_train() -> None:
    model = _model()
    assert not any(p.requires_grad for p in model.backbone.parameters())
    heads = [p for p in model.parameters() if p.requires_grad]
    assert heads
    model.train()
    assert not model.backbone.training  # frozen means eval mode too


def test_a_reloaded_model_gives_the_same_outputs(
    tmp_path: pathlib.Path,
) -> None:
    model = _model()
    model.save(tmp_path / "hydra")
    again = hydra.HydraNet.load(tmp_path / "hydra").eval()
    x = _input()
    with torch.inference_mode():
        a, b = model(x), again(x)
    for name in ("floor", "log_depth", "heatmap", "size", "offset"):
        torch.testing.assert_close(getattr(a, name), getattr(b, name))


def test_the_predictor_answers_all_three_tasks_from_one_forward(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _model().save(tmp_path / "hydra")
    predictor = hydra.Hydra(tmp_path / "hydra")
    calls: list[int] = []
    real = predictor._model.forward

    def counted(x: torch.Tensor) -> hydra.Heads:
        calls.append(1)
        heads: hydra.Heads = real(x)
        return heads

    monkeypatch.setattr(predictor._model, "forward", counted)
    rgb = np.zeros((300, 401, 3), np.uint8)
    floor, disparity, found = predictor.predict_all(rgb)
    assert len(calls) == 1
    assert floor.shape == (300, 401) and floor.dtype == bool
    assert disparity.shape == (300, 401) and (disparity > 0).all()
    for d in found:
        assert d.category in boxes.CATEGORIES
        assert 0 <= d.x <= 401 and 0 <= d.y <= 300
