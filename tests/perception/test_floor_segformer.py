"""Tests for minifsd.perception.floor_segformer (E2a).

Preprocessing runs everywhere. The model tests need torch, so they run in
the cv environment and skip elsewhere:

    pixi run -e cv pytest tests/perception/test_floor_segformer.py
"""

import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.perception import floor_segformer as fs


@settings(deadline=None, max_examples=20)
@given(st.integers(60, 900), st.integers(60, 1200))
def test_preprocess_gives_the_training_size_for_any_photo(
    h: int, w: int
) -> None:
    x = fs.preprocess(np.zeros((h, w, 3), np.uint8))
    assert x.shape == (1, 3, fs.INPUT_HEIGHT, fs.INPUT_WIDTH)
    assert x.dtype == np.float32


def test_preprocess_normalises_with_imagenet_statistics() -> None:
    rgb = np.full((240, 320, 3), 255, np.uint8)
    x = fs.preprocess(rgb)
    expected = (1.0 - np.array(fs.MEAN)) / np.array(fs.STD)
    np.testing.assert_allclose(x[0, :, 0, 0], expected, rtol=1e-5)


def test_mask_from_logits_follows_the_larger_logit_at_full_size() -> None:
    # Floor (class 1) wins on the bottom half of a 4 x 4 logit map.
    logits = np.zeros((2, 4, 4), np.float32)
    logits[1, 2:] = 5.0
    logits[0, :2] = 5.0
    mask = fs.mask_from_logits(logits, 40, 60)
    assert mask.shape == (40, 60) and mask.dtype == bool
    assert mask[-5:].all() and not mask[:5].any()


def test_a_saved_model_predicts_a_mask_the_photo_size(
    tmp_path: pathlib.Path,
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from minifsd.train import segformer

    segformer.build_model(None).save_pretrained(tmp_path / "model")
    model = fs.FloorSegformer(tmp_path / "model")
    mask = model.predict(np.zeros((300, 401, 3), np.uint8))
    assert mask.shape == (300, 401) and mask.dtype == bool
