"""Tests for minifsd.perception.floor_classical."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from minifsd.eval import metrics
from minifsd.perception import floor_classical as fc

H, W = 120, 160
WALL = (200, 200, 180)
FLOOR = (90, 70, 50)
BOX = (30, 90, 160)


def _scene(box: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A wall above a floor, with mild noise. Returns (rgb, true floor)."""
    rng = np.random.default_rng(0)
    rgb = np.empty((H, W, 3), np.float64)
    rgb[: H // 2] = WALL
    rgb[H // 2 :] = FLOOR
    floor = np.zeros((H, W), bool)
    floor[H // 2 :] = True
    if box:  # an obstacle standing on the floor, off to one side
        rgb[70:100, 20:60] = BOX
        floor[70:100, 20:60] = False
    rgb += rng.normal(0, 1.5, rgb.shape)  # sensor noise
    return np.clip(rgb, 0, 255).astype(np.uint8), floor


def test_finds_floor_below_a_wall() -> None:
    rgb, floor = _scene()
    assert metrics.floor_iou(fc.floor_mask(rgb), floor) > 0.95


def test_leaves_out_an_obstacle_on_the_floor() -> None:
    rgb, floor = _scene(box=True)
    assert metrics.floor_iou(fc.floor_mask(rgb), floor) > 0.9


def test_seed_on_an_edge_gives_empty_mask() -> None:
    rgb, _ = _scene()
    row, col = fc.seed_pixel(H, W, fc.Params())
    rgb[row - 2 : row + 3, :] = 255  # bright stripe through the seed
    rgb[row, :] = 0
    assert not fc.floor_mask(rgb).any()


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((10, 10), np.uint8),  # no channels
        np.zeros((10, 10, 3), np.float32),  # wrong dtype
        np.zeros((5, 10, 3), np.uint8),  # too short for the seed
    ],
)
def test_rejects_bad_images(rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fc.floor_mask(rgb)


# --- Properties -------------------------------------------------------------

IMAGES = hnp.arrays(
    np.uint8,
    st.tuples(st.integers(8, 32), st.integers(8, 32), st.just(3)),
)
COLOURS = st.tuples(*[st.integers(0, 255)] * 3)


@given(IMAGES)
def test_mask_is_empty_or_contains_seed(rgb: np.ndarray) -> None:
    mask = fc.floor_mask(rgb)
    h, w = rgb.shape[:2]
    assert mask.shape == (h, w) and mask.dtype == bool
    row, col = fc.seed_pixel(h, w, fc.Params())
    assert not mask.any() or mask[row, col]


@given(IMAGES, st.data())
def test_nothing_above_the_horizon(
    rgb: np.ndarray, data: st.DataObject
) -> None:
    horizon = data.draw(st.integers(0, rgb.shape[0] - 1))
    mask = fc.floor_mask(rgb, fc.Params(horizon_row=horizon))
    assert not mask[: horizon + 1].any()


@given(COLOURS, st.integers(8, 32), st.integers(8, 32))
def test_plain_image_is_all_floor(
    colour: tuple[int, int, int], h: int, w: int
) -> None:
    # No edges and no colour change: the fill must reach every pixel.
    rgb = np.full((h, w, 3), colour, np.uint8)
    assert fc.floor_mask(rgb).all()
