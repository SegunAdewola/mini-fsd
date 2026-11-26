"""Tests for minifsd.perception.detect_onnx: exported detectors (E1).

Letterboxing and box decoding are numpy and run everywhere. The model test
needs onnxruntime and an exported file, so it runs in the cv env once
data/models/detect-yolo26n/best.onnx is there:

    pixi run -e cv pytest tests/perception/test_detect_onnx.py
"""

import importlib.util
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.data import boxes
from minifsd.perception import detect_onnx

SIZE = 320


@settings(deadline=None, max_examples=30)
@given(st.integers(40, 1200), st.integers(40, 1600))
def test_letterbox_fits_any_photo_into_the_square_unstretched(
    h: int, w: int
) -> None:
    x, frame = detect_onnx.letterbox(np.zeros((h, w, 3), np.uint8), SIZE)
    assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
    # The long side fills the square; the short side keeps the aspect ratio.
    assert max(h, w) * frame.scale == pytest.approx(SIZE, abs=1)
    assert frame.pad_x >= 0 and frame.pad_y >= 0
    assert frame.pad_x == 0 or frame.pad_y == 0


def test_letterbox_pads_with_grey_and_scales_to_0_1() -> None:
    rgb = np.full((240, 320, 3), 255, np.uint8)
    x, frame = detect_onnx.letterbox(rgb, SIZE)
    assert (frame.pad_x, frame.pad_y) == (0, 40)
    assert x[0, :, 0, 0] == pytest.approx([114 / 255] * 3)  # padding
    assert x[0, :, SIZE // 2, SIZE // 2] == pytest.approx([1.0] * 3)


@settings(deadline=None, max_examples=30)
@given(
    st.integers(100, 1200),
    st.integers(100, 1600),
    st.floats(0.0, 0.5),
    st.floats(0.0, 0.5),
    st.floats(0.1, 0.5),
    st.floats(0.1, 0.5),
)
def test_a_box_maps_back_to_the_photo_it_came_from(
    h: int, w: int, fx: float, fy: float, fw: float, fh: float
) -> None:
    # A box in the photo, pushed through the letterbox by hand, must come
    # back where it was.
    _, frame = detect_onnx.letterbox(np.zeros((h, w, 3), np.uint8), SIZE)
    x0, y0, bw, bh = fx * w, fy * h, fw * w, fh * h
    row = np.array(
        [
            [
                x0 * frame.scale + frame.pad_x,
                y0 * frame.scale + frame.pad_y,
                (x0 + bw) * frame.scale + frame.pad_x,
                (y0 + bh) * frame.scale + frame.pad_y,
                0.9,
                boxes.CATEGORIES.index("door"),
            ]
        ],
        np.float32,
    )
    (d,) = detect_onnx.to_detections(row, frame, min_score=0.0)
    assert d.category == "door" and d.score == pytest.approx(0.9)
    assert (d.x, d.y, d.w, d.h) == pytest.approx((x0, y0, bw, bh), abs=0.5)


def test_low_scores_are_dropped_and_boxes_clipped_to_the_photo() -> None:
    _, frame = detect_onnx.letterbox(np.zeros((240, 320, 3), np.uint8), SIZE)
    rows = np.array(
        [
            [-10, 30, 50, 100, 0.8, 0],  # pokes out of the left edge
            [10, 50, 50, 100, 0.0005, 1],  # below min_score
        ],
        np.float32,
    )
    found = detect_onnx.to_detections(rows, frame, min_score=0.001)
    assert len(found) == 1
    assert found[0].category == "obstacle"
    assert found[0].x == 0.0


def test_rows_of_the_wrong_shape_are_refused() -> None:
    _, frame = detect_onnx.letterbox(np.zeros((240, 320, 3), np.uint8), SIZE)
    with pytest.raises(ValueError, match="end-to-end"):
        detect_onnx.to_detections(np.zeros((8, 84), np.float32), frame, 0.0)


MODEL = pathlib.Path("data/models/detect-yolo26n/best.onnx")


@pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None or not MODEL.is_file(),
    reason="needs onnxruntime (cv env) and the exported E1a model",
)
def test_the_exported_model_returns_detections_in_photo_pixels() -> None:
    model = detect_onnx.OnnxDetector(MODEL, threads=2)
    found = model.predict(np.zeros((480, 640, 3), np.uint8))
    assert isinstance(found, list)
    for d in found:
        assert d.category in boxes.CATEGORIES
        assert 0 <= d.x <= 640 and 0 <= d.y <= 480


# RF-DETR (E1b): stretched to a square, ImageNet-normalised; outputs
# normalised cxcywh boxes and per-class logits.


def test_stretch_resizes_to_the_square_and_normalises() -> None:
    x = detect_onnx.stretch(np.full((240, 320, 3), 255, np.uint8), SIZE)
    assert x.shape == (1, 3, SIZE, SIZE) and x.dtype == np.float32
    expected = (1.0 - np.array(detect_onnx.IMAGENET_MEAN)) / np.array(
        detect_onnx.IMAGENET_STD
    )
    assert x[0, :, 5, 5] == pytest.approx(expected, rel=1e-5)


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_detr_outputs_map_to_photo_pixels_and_categories() -> None:
    q = 3
    dets = np.zeros((q, 4), np.float32)
    logits = np.full((q, len(boxes.CATEGORIES)), -10.0, np.float32)
    dets[0] = [0.5, 0.5, 0.2, 0.4]  # centred box, 20% wide, 40% tall
    logits[0, boxes.CATEGORIES.index("door")] = _logit(0.9)
    (d,) = detect_onnx.decode_detr(dets, logits, 640, 480, min_score=0.5)
    assert d.category == "door" and d.score == pytest.approx(0.9, abs=1e-5)
    assert (d.x, d.y, d.w, d.h) == pytest.approx((256, 144, 128, 192))


def test_one_query_can_be_two_classes_as_rf_detr_scores_them() -> None:
    # Per-class sigmoids, not a softmax: a query may score high on two.
    dets = np.array([[0.5, 0.5, 0.2, 0.2]], np.float32)
    logits = np.full((1, len(boxes.CATEGORIES)), -10.0, np.float32)
    logits[0, 0] = _logit(0.8)
    logits[0, 3] = _logit(0.6)
    found = detect_onnx.decode_detr(dets, logits, 100, 100, min_score=0.5)
    assert [(d.category, round(d.score, 3)) for d in found] == [
        ("obstacle", 0.8),
        ("person", 0.6),
    ]


def test_a_class_slot_beyond_our_categories_is_ignored() -> None:
    dets = np.array([[0.5, 0.5, 0.2, 0.2]], np.float32)
    logits = np.full((1, len(boxes.CATEGORIES) + 1), -10.0, np.float32)
    logits[0, -1] = _logit(0.99)  # an extra slot some exports carry
    assert detect_onnx.decode_detr(dets, logits, 100, 100, 0.5) == []
