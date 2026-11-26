"""Tests for minifsd.perception.centernet: box targets and decoding (H1)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.perception import centernet as cn

H, W = 238, 322  # HydraNet's input
GRID = (H // cn.STRIDE, W // cn.STRIDE)
N_CLASSES = 4


@st.composite
def _box(draw: st.DrawFn) -> cn.Box:
    w = draw(st.floats(8, 150))
    h = draw(st.floats(8, 150))
    x = draw(st.floats(0, W - w - 1))
    y = draw(st.floats(0, H - h - 1))
    return cn.Box(draw(st.integers(0, N_CLASSES - 1)), x, y, w, h)


@settings(deadline=None, max_examples=60)
@given(_box())
def test_a_box_survives_encoding_and_decoding(box: cn.Box) -> None:
    t = cn.encode([box], GRID, N_CLASSES)
    (found, *_) = cn.decode(t.heatmap, t.size, t.offset, min_score=0.5)
    assert found.cls == box.cls
    assert found.score == pytest.approx(1.0)
    assert (found.x, found.y, found.w, found.h) == pytest.approx(
        (box.x, box.y, box.w, box.h), abs=1e-3
    )


@settings(deadline=None, max_examples=30)
@given(_box())
def test_the_heatmap_peaks_at_one_on_the_centre_cell_only(box: cn.Box) -> None:
    t = cn.encode([box], GRID, N_CLASSES)
    row = int((box.y + box.h / 2) / cn.STRIDE)
    col = int((box.x + box.w / 2) / cn.STRIDE)
    assert t.heatmap[box.cls, row, col] == 1.0
    assert (t.heatmap[box.cls] == 1.0).sum() == 1
    assert t.heatmap.max(initial=0) <= 1.0 and t.heatmap.min(initial=1) >= 0.0
    others = np.delete(t.heatmap, box.cls, axis=0)
    assert not others.any()
    assert t.mask[row, col] and t.mask.sum() == 1


def test_two_classes_on_one_spot_both_decode() -> None:
    # A person standing in a doorway: same centre, two classes.
    boxes = [cn.Box(1, 100, 50, 60, 120), cn.Box(3, 100, 50, 60, 120)]
    t = cn.encode(boxes, GRID, N_CLASSES)
    found = cn.decode(t.heatmap, t.size, t.offset, min_score=0.5)
    assert sorted(f.cls for f in found) == [1, 3]


def test_two_boxes_of_one_class_far_apart_both_decode() -> None:
    boxes = [cn.Box(0, 10, 10, 40, 40), cn.Box(0, 250, 150, 50, 60)]
    t = cn.encode(boxes, GRID, N_CLASSES)
    found = cn.decode(t.heatmap, t.size, t.offset, min_score=0.5)
    assert len(found) == 2
    assert sorted(round(f.x) for f in found) == [10, 250]


def test_decoding_keeps_the_best_first_and_drops_low_scores() -> None:
    heatmap = np.zeros((N_CLASSES, *GRID), np.float32)
    heatmap[0, 5, 5] = 0.9
    heatmap[2, 20, 30] = 0.3
    heatmap[1, 10, 10] = 0.0005
    size = np.full((2, *GRID), 3.0, np.float32)
    offset = np.zeros((2, *GRID), np.float32)
    found = cn.decode(heatmap, size, offset, min_score=0.001)
    assert [(f.cls, f.score) for f in found] == [
        (0, pytest.approx(0.9)),
        (2, pytest.approx(0.3)),
    ]


def test_a_box_too_small_to_have_a_centre_cell_is_still_encoded() -> None:
    # A far-away sign of 4 x 3 px still gets its one peak.
    t = cn.encode([cn.Box(2, 30, 30, 4, 3)], GRID, N_CLASSES)
    assert t.heatmap[2].max() == 1.0
