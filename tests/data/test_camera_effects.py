"""Tests for minifsd.data.camera_effects: what a real camera does to a frame."""

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.data import camera_effects as fx

SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


def _image(seed: int, h: int = 48, w: int = 64) -> np.ndarray:
    """A smooth image with some edges, like a render: blocks plus a ramp."""
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (h // 8, w // 8, 3)).astype(np.float32)
    big = cv2.resize(blocks, (w, h), interpolation=cv2.INTER_NEAREST)
    ramp = np.linspace(0, 60, w, dtype=np.float32)[None, :, None]
    return np.asarray(np.clip(big * 0.75 + ramp, 0, 255).astype(np.uint8))


def _sharpness(rgb: np.ndarray) -> float:
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


@settings(deadline=None)
@given(SEEDS, SEEDS)
def test_keeps_shape_and_dtype_for_any_sampled_effects(
    image_seed: int, seed: int
) -> None:
    rgb = _image(image_seed)
    rng = np.random.default_rng(seed)
    out = fx.apply(rgb, fx.sample_effects(rng), rng)
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


@settings(deadline=None)
@given(SEEDS)
def test_the_same_rng_gives_the_same_output(seed: int) -> None:
    rgb = _image(0)
    a_rng, b_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    a = fx.apply(rgb, fx.sample_effects(a_rng), a_rng)
    b = fx.apply(rgb, fx.sample_effects(b_rng), b_rng)
    np.testing.assert_array_equal(a, b)


@given(SEEDS)
def test_the_default_does_nothing(seed: int) -> None:
    rgb = _image(seed)
    out = fx.apply(rgb, fx.Effects(), np.random.default_rng(0))
    np.testing.assert_array_equal(out, rgb)


def test_does_not_change_its_input() -> None:
    rgb = _image(1)
    before = rgb.copy()
    fx.apply(rgb, fx.Effects(noise_sigma=8.0), np.random.default_rng(0))
    np.testing.assert_array_equal(rgb, before)


@pytest.mark.parametrize(
    "effects",
    [
        fx.Effects(blur_sigma=1.5),
        fx.Effects(motion_px=6.0, motion_angle=0.7),
    ],
    ids=["defocus", "motion"],
)
def test_blur_softens_edges(effects: fx.Effects) -> None:
    rgb = _image(2)
    out = fx.apply(rgb, effects, np.random.default_rng(0))
    assert _sharpness(out) < 0.5 * _sharpness(rgb)


def test_motion_blur_smears_along_its_angle_only() -> None:
    # A horizontal smear blurs vertical edges and leaves horizontal ones.
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[:, 20:] = 200  # one vertical edge
    out = fx.apply(rgb, fx.Effects(motion_px=7.0), np.random.default_rng(0))
    assert (out[20, 14:26, 0] > 0).sum() > (rgb[20, 14:26, 0] > 0).sum()
    rgb_h = np.zeros((40, 40, 3), np.uint8)
    rgb_h[20:] = 200  # one horizontal edge
    out_h = fx.apply(rgb_h, fx.Effects(motion_px=7.0), np.random.default_rng(0))
    np.testing.assert_array_equal(out_h[5:35, 10:30], rgb_h[5:35, 10:30])


def test_noise_has_the_asked_for_spread() -> None:
    grey = np.full((200, 200, 3), 128, np.uint8)
    out = fx.apply(grey, fx.Effects(noise_sigma=6.0), np.random.default_rng(0))
    diff = out.astype(float) - grey
    assert abs(diff.mean()) < 0.3
    assert diff.std() == pytest.approx(6.0, rel=0.05)


@pytest.mark.parametrize(
    ("effects", "brighter"),
    [
        (fx.Effects(exposure=1.4), True),
        (fx.Effects(exposure=0.6), False),
        (fx.Effects(gamma=0.8), True),  # out = in ** gamma, on [0, 1]
        (fx.Effects(gamma=1.25), False),
    ],
)
def test_exposure_and_gamma_move_brightness_the_right_way(
    effects: fx.Effects, brighter: bool
) -> None:
    rgb = _image(3)
    out = fx.apply(rgb, effects, np.random.default_rng(0))
    assert (out.mean() > rgb.mean()) == brighter


def test_gamma_keeps_black_and_white() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:2] = 255
    for g in (0.8, 1.25):
        out = fx.apply(rgb, fx.Effects(gamma=g), np.random.default_rng(0))
        np.testing.assert_array_equal(out, rgb)


def test_low_jpeg_quality_leaves_artefacts() -> None:
    rgb = _image(4)
    low = fx.apply(rgb, fx.Effects(jpeg_quality=30), np.random.default_rng(0))
    high = fx.apply(rgb, fx.Effects(jpeg_quality=95), np.random.default_rng(0))
    err_low = np.abs(low.astype(float) - rgb).mean()
    err_high = np.abs(high.astype(float) - rgb).mean()
    assert err_low > err_high > 0


@given(SEEDS)
def test_sampled_effects_stay_in_range(seed: int) -> None:
    for e in (
        fx.sample_effects(np.random.default_rng([seed, i])) for i in range(20)
    ):
        assert 0.0 <= e.blur_sigma <= fx.BLUR_SIGMA_MAX
        assert 0.0 <= e.motion_px <= fx.MOTION_PX_MAX
        assert 0.0 <= e.noise_sigma <= fx.NOISE_SIGMA_MAX
        assert fx.EXPOSURE_RANGE[0] <= e.exposure <= fx.EXPOSURE_RANGE[1]
        assert fx.GAMMA_RANGE[0] <= e.gamma <= fx.GAMMA_RANGE[1]
        lo, hi = fx.JPEG_QUALITY_RANGE
        assert e.jpeg_quality is None or lo <= e.jpeg_quality <= hi


def test_some_sampled_frames_are_clean() -> None:
    # Not every phone photo is blurred or compressed hard; each effect is
    # on in only part of the frames, so models also see sharp images.
    effects = [fx.sample_effects(np.random.default_rng(i)) for i in range(500)]
    blurred = np.mean([e.blur_sigma > 0 for e in effects])
    smeared = np.mean([e.motion_px > 0 for e in effects])
    assert abs(blurred - fx.BLUR_CHANCE) < 0.07
    assert abs(smeared - fx.MOTION_CHANCE) < 0.07


@pytest.mark.parametrize(
    "bad",
    [
        {"blur_sigma": -1.0},
        {"motion_px": -2.0},
        {"noise_sigma": -0.5},
        {"exposure": 0.0},
        {"gamma": 0.0},
        {"jpeg_quality": 0},
        {"jpeg_quality": 101},
    ],
)
def test_rejects_effects_that_make_no_sense(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        fx.Effects(**bad)  # type: ignore[arg-type]
