"""Tests for minifsd.train.hydra_loss: HydraNet-lite's losses (H2).

Need torch: pixi run -e cv pytest tests/train/test_hydra_loss.py
"""

from __future__ import annotations  # names below exist only with torch

import importlib.util

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# A skip mark, not a module-level importorskip (see test_segformer.py).
HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="needs torch (cv env)")
if HAS_TORCH:
    import torch

    from minifsd.train import hydra_loss as hl

BIG = 30.0  # a logit that is, for the loss, certain


def _heatmaps() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(2, 4, 6, 8)
    target[0, 1, 2, 3] = 1.0
    target[0, 1, 2, 4] = 0.6  # Gaussian shoulder: a near miss, not a hit
    target[1, 3, 4, 5] = 1.0
    logits = torch.full_like(target, -BIG)
    logits[target == 1.0] = BIG
    return logits, target


def test_focal_loss_is_zero_for_a_perfect_heatmap() -> None:
    logits, target = _heatmaps()
    every_class = torch.ones(2, 4, dtype=torch.bool)
    assert float(hl.focal_loss(logits, target, every_class)) == pytest.approx(
        0.0, abs=1e-4
    )


def test_a_masked_class_adds_nothing() -> None:
    # COCO images: door (1) and sign (2) are unlabeled, not absent.
    logits, target = _heatmaps()
    mask = torch.ones(2, 4, dtype=torch.bool)
    mask[1, 1:3] = False
    before = float(hl.focal_loss(logits, target, mask))
    logits[1, 1:3] = BIG  # confident doors and signs on image 1
    assert float(hl.focal_loss(logits, target, mask)) == pytest.approx(before)
    every = torch.ones(2, 4, dtype=torch.bool)
    assert float(hl.focal_loss(logits, target, every)) > before + 1.0


def test_size_loss_counts_only_centre_cells() -> None:
    target = torch.zeros(1, 2, 4, 4)
    target[0, :, 1, 1] = torch.tensor([3.0, 5.0])
    centres = torch.zeros(1, 4, 4, dtype=torch.bool)
    centres[0, 1, 1] = True
    pred = torch.full_like(target, 99.0)  # wrong everywhere but the centre
    pred[0, :, 1, 1] = target[0, :, 1, 1] + torch.tensor([1.0, -1.0])
    assert float(hl.masked_l1(pred, target, centres)) == pytest.approx(1.0)
    assert float(hl.masked_l1(pred, target, centres & False)) == 0.0


def test_floor_loss_skips_images_without_floor_labels() -> None:
    labels = torch.zeros(2, 8, 8, dtype=torch.long)
    labels[:, 4:] = 1
    logits = torch.zeros(2, 2, 4, 4)
    logits[0, 1, 2:] = BIG
    logits[0, 0, :2] = BIG
    has_floor = torch.tensor([True, False])
    loss = hl.floor_loss(logits, labels, has_floor)
    assert float(loss) < 0.2  # image 0 is near perfect; image 1 is ignored
    none = hl.floor_loss(logits, labels, torch.tensor([False, False]))
    assert float(none) == 0.0


@settings(deadline=None, max_examples=20)
@given(st.floats(0.1, 10.0))
def test_depth_loss_ignores_a_global_scale(scale: float) -> None:
    # The harness fits scale; the sim camera's isn't a phone's.
    gen = torch.Generator().manual_seed(0)
    depth = torch.rand(2, 6, 8, generator=gen) * 5 + 0.5
    has_depth = torch.tensor([True, True])
    log_pred = torch.log(depth * scale).unsqueeze(1)
    assert float(hl.depth_loss(log_pred, depth, has_depth)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_depth_loss_skips_empty_pixels_and_unlabeled_images() -> None:
    depth = torch.full((2, 4, 4), 2.0)
    depth[0, 0, 0] = 0.0  # no reading: must not count
    log_pred = torch.log(depth.clamp(min=1e-3)).unsqueeze(1)
    log_pred[1] = 5.0  # image 1 is wrong, but has no depth label
    loss = hl.depth_loss(log_pred, depth, torch.tensor([True, False]))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
