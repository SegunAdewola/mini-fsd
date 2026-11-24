"""Tests for minifsd.perception.depth_anything.

Preprocessing is plain numpy and OpenCV and runs everywhere, CI included.
The model tests need the cv env (torch) and the weights in data/models; they
skip elsewhere. Run them with:
    pixi run -e cv pytest tests/perception/test_depth_anything.py
"""

import pathlib

import numpy as np
import pytest

from minifsd.perception import depth_anything

MODEL_DIR = pathlib.Path("data/models/depth-anything-v2-small")
NYU = pathlib.Path("data/nyuv2")


@pytest.mark.parametrize(
    ("h", "w"), [(426, 560), (480, 640), (240, 320), (560, 426)]
)
def test_input_size_keeps_aspect_and_is_a_multiple_of_14(
    h: int, w: int
) -> None:
    x = depth_anything.preprocess(np.zeros((h, w, 3), dtype=np.uint8))
    _, c, hh, ww = x.shape
    assert c == 3
    assert hh % 14 == 0 and ww % 14 == 0
    assert min(hh, ww) >= depth_anything.SHORT_SIDE  # never shrink below
    assert min(hh, ww) < depth_anything.SHORT_SIDE + 14  # nor overshoot
    assert ww / hh == pytest.approx(w / h, rel=0.03)


def test_normalises_with_imagenet_statistics() -> None:
    grey = np.full((240, 320, 3), 128, dtype=np.uint8)
    x = depth_anything.preprocess(grey)
    assert x.dtype == np.float32
    expected = (128 / 255 - np.array(depth_anything.MEAN)) / np.array(
        depth_anything.STD
    )
    np.testing.assert_allclose(x[0, :, 10, 10], expected, rtol=1e-5)


def test_rejects_non_rgb_input() -> None:
    with pytest.raises(ValueError):
        depth_anything.preprocess(np.zeros((240, 320), dtype=np.uint8))


def _model() -> depth_anything.DepthAnything:
    pytest.importorskip("torch")
    if not (MODEL_DIR / "model.safetensors").is_file():
        pytest.skip(f"weights not in {MODEL_DIR}")
    return depth_anything.DepthAnything(MODEL_DIR, device="cpu")


def test_disparity_has_the_input_size() -> None:
    model = _model()
    rgb = np.random.default_rng(0).integers(0, 255, (240, 320, 3), np.uint8)
    disparity = model.predict(rgb)
    assert disparity.shape == (240, 320)
    assert disparity.dtype == np.float32
    assert np.isfinite(disparity).all()


def test_zero_shot_absrel_on_20_nyuv2_test_images() -> None:
    model = _model()
    mat = NYU / "nyu_depth_v2_labeled.mat"
    if not mat.is_file():
        pytest.skip("NYUv2 not downloaded")
    from minifsd.data import nyuv2
    from minifsd.eval import depth_align, metrics

    test = nyuv2.load_split(NYU / "splits.mat")["test"][:20]
    scores = []
    with nyuv2.NyuV2(mat) as data:
        for i in test:
            s = nyuv2.crop(data[i])
            disparity = model.predict(s.rgb)
            depth = depth_align.fit(disparity, s.depth).to_depth(disparity)
            scores.append(metrics.abs_rel(depth, s.depth))
    # Loose bound: the paper reports about 0.05 for this model with
    # per-image alignment; 0.15 catches a broken pipeline, not a bad model.
    assert float(np.mean(scores)) < 0.15
