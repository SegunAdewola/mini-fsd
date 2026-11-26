"""Tests for minifsd.demo.overlay: drawing model outputs onto photos."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from minifsd.demo import overlay
from minifsd.eval import detection

SHAPES = st.tuples(st.integers(1, 24), st.integers(1, 24))


@st.composite
def image_and_mask(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray]:
    h, w = draw(SHAPES)
    rgb = draw(hnp.arrays(np.uint8, (h, w, 3)))
    mask = draw(hnp.arrays(np.bool_, (h, w)))
    return rgb, mask


@given(image_and_mask())
def test_pixels_off_the_floor_are_unchanged(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    rgb, mask = pair
    out = overlay.draw_floor(rgb, mask)
    np.testing.assert_array_equal(out[~mask], rgb[~mask])


@given(image_and_mask())
def test_floor_pixels_are_blended_with_the_floor_colour(
    pair: tuple[np.ndarray, np.ndarray],
) -> None:
    rgb, mask = pair
    out = overlay.draw_floor(rgb, mask)
    a = overlay.FLOOR_ALPHA
    expected = (1 - a) * rgb[mask] + a * np.array(overlay.FLOOR_RGB)
    # Off by at most 1 from rounding to uint8.
    np.testing.assert_allclose(out[mask], expected, atol=1.0)


def test_keeps_shape_and_dtype_and_leaves_the_input_alone() -> None:
    rgb = np.full((4, 6, 3), 100, np.uint8)
    before = rgb.copy()
    out = overlay.draw_floor(rgb, np.ones((4, 6), bool))
    assert out.shape == rgb.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(rgb, before)


def test_rejects_a_mask_of_another_size() -> None:
    with pytest.raises(ValueError):
        overlay.draw_floor(np.zeros((4, 6, 3), np.uint8), np.ones((4, 5), bool))


def test_a_box_is_outlined_and_its_inside_left_alone() -> None:
    rgb = np.zeros((60, 80, 3), np.uint8)
    box = detection.Detection("person", 20.0, 10.0, 30.0, 40.0, 0.9)
    out = overlay.draw_boxes(rgb, [box])
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert out[30, 20].any()  # on the left edge
    assert not out[30, 35].any()  # well inside, away from the label
    assert not rgb.any()  # the input is untouched


def test_each_class_has_its_own_colour() -> None:
    rgb = np.zeros((60, 80, 3), np.uint8)
    colours = {
        tuple(
            overlay.draw_boxes(
                rgb, [detection.Detection(c, 20.0, 10.0, 30.0, 40.0, 0.9)]
            )[30, 20]
        )
        for c in overlay.BOX_RGB
    }
    assert len(colours) == len(overlay.BOX_RGB)
