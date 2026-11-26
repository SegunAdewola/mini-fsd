"""Exported detectors run with ONNX Runtime (E1, ADR 0008).

Detectors are trained on Colab and exported to ONNX; on the M1 they run
here, with no training library loaded. That keeps Ultralytics' code off the
M1 and times both E1 detectors in the same runtime.

Input: the photo letterboxed into a size x size square (scaled to fit, then
padded with grey 114 around the short side, centred), RGB, scaled to 0..1:
what Ultralytics trains with.

Output: an end-to-end export, which needs no non-maximum suppression: one
(1, N, 6) tensor, each row x1, y1, x2, y2 in letterboxed pixels, a score
and a class index into minifsd.data.boxes.CATEGORIES. YOLO26 exports this
way by default; the first real export confirms it.

RF-DETR (E1b) differs, as read from rfdetr 1.12.1's own export code: the
photo is stretched to the square (no letterbox), bilinear, ImageNet
normalised; the outputs are "dets" (1, Q, 4), boxes as centre x, centre y,
width and height in fractions of the input, and "labels" (1, Q, C),
per-class logits. Each class has its own sigmoid, so one query can be two
classes; the top query and class pairs are kept. Class c is
CATEGORIES[c] (RF-DETR numbers classes by their place in the train file).

onnxruntime is imported inside OnnxDetector, so the rest of this module
loads in the core env.
"""

import dataclasses
import pathlib

import cv2
import numpy as np

from minifsd.data import boxes
from minifsd.eval import detection

PAD_VALUE = 114  # Ultralytics' letterbox grey
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # RF-DETR's input normalisation
IMAGENET_STD = (0.229, 0.224, 0.225)
DETR_SELECT = 300  # query and class pairs kept, RF-DETR's num_select
KINDS = ("end2end", "detr")
MIN_SCORE = 0.001  # keep nearly everything: mAP needs the low-score tail
ROW_WIDTH = 6


@dataclasses.dataclass(frozen=True)
class Frame:
    """How a photo sits inside the letterboxed square."""

    scale: float  # photo pixels to square pixels
    pad_x: float  # grey added on the left
    pad_y: float  # grey added on top
    width: int  # the photo's own size
    height: int


def letterbox(rgb: np.ndarray, size: int) -> tuple[np.ndarray, Frame]:
    """Fits a photo into a size x size square, as the detector was trained.

    Returns:
        The (1, 3, size, size) float32 input in 0..1, and where the photo
        sits in it.
    """
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = cv2.resize(
        rgb,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    square = np.full((size, size, 3), PAD_VALUE, np.uint8)
    square[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    x = square.astype(np.float32).transpose(2, 0, 1)[None] / np.float32(255)
    return np.ascontiguousarray(x), Frame(scale, pad_x, pad_y, w, h)


def to_detections(
    rows: np.ndarray, frame: Frame, min_score: float = MIN_SCORE
) -> list[detection.Detection]:
    """End-to-end output rows to boxes in the photo's pixels.

    Args:
        rows: (N, 6) x1, y1, x2, y2, score, class in letterboxed pixels.
        frame: From letterbox().
        min_score: Rows scoring below this are dropped.

    Raises:
        ValueError: If rows aren't (N, 6), e.g. an export that still needs
            non-maximum suppression.
    """
    if rows.ndim != 2 or rows.shape[1] != ROW_WIDTH:
        raise ValueError(
            f"expected end-to-end rows (N, {ROW_WIDTH}), got {rows.shape}"
        )
    found = []
    for x1, y1, x2, y2, score, cls in rows.astype(np.float64):
        if score < min_score:
            continue
        # Undo the padding and the scale, then keep the box in the photo.
        left = min(max((x1 - frame.pad_x) / frame.scale, 0.0), frame.width)
        top = min(max((y1 - frame.pad_y) / frame.scale, 0.0), frame.height)
        right = min(max((x2 - frame.pad_x) / frame.scale, 0.0), frame.width)
        bottom = min(max((y2 - frame.pad_y) / frame.scale, 0.0), frame.height)
        found.append(
            detection.Detection(
                category=boxes.CATEGORIES[int(cls)],
                x=left,
                y=top,
                w=right - left,
                h=bottom - top,
                score=float(score),
            )
        )
    return found


def stretch(rgb: np.ndarray, size: int) -> np.ndarray:
    """A photo resized to size x size, ImageNet-normalised, for RF-DETR."""
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / np.float32(255)
    x = (scaled - np.array(IMAGENET_MEAN, np.float32)) / np.array(
        IMAGENET_STD, np.float32
    )
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def decode_detr(
    dets: np.ndarray,
    logits: np.ndarray,
    width: int,
    height: int,
    min_score: float = MIN_SCORE,
    select: int = DETR_SELECT,
) -> list[detection.Detection]:
    """RF-DETR's outputs for one image to boxes in its pixels.

    Args:
        dets: (Q, 4) centre x, centre y, width, height as fractions.
        logits: (Q, C) per-class logits; slots past CATEGORIES are ignored.
        width: The photo's width in pixels.
        height: The photo's height in pixels.
        min_score: Pairs scoring below this are dropped.
        select: How many of the best query and class pairs to keep.

    Returns:
        Detections, best score first.
    """
    scores = 1.0 / (1.0 + np.exp(-logits[:, : len(boxes.CATEGORIES)]))
    n_classes = scores.shape[1]
    flat = scores.ravel()
    top = np.argsort(-flat, kind="stable")[:select]
    found = []
    for index in top:
        score = float(flat[index])
        if score < min_score:
            break
        query, cls = divmod(int(index), n_classes)
        cx, cy, w, h = dets[query].astype(np.float64)
        left = min(max((cx - w / 2) * width, 0.0), width)
        top_y = min(max((cy - h / 2) * height, 0.0), height)
        right = min(max((cx + w / 2) * width, 0.0), width)
        bottom = min(max((cy + h / 2) * height, 0.0), height)
        found.append(
            detection.Detection(
                category=boxes.CATEGORIES[cls],
                x=left,
                y=top_y,
                w=right - left,
                h=bottom - top_y,
                score=score,
            )
        )
    return found


class OnnxDetector:
    """An exported end-to-end detector."""

    def __init__(
        self,
        path: pathlib.Path,
        threads: int,
        kind: str = "end2end",
        size: int | None = None,
    ) -> None:
        """Opens the model on the CPU.

        Args:
            path: The .onnx file.
            threads: ONNX Runtime's thread cap (the harness's budget).
            kind: "end2end" (YOLO26) or "detr" (RF-DETR); see the module
                docstring.
            size: Input side in pixels; read from the model if None.

        Raises:
            ValueError: For an unknown kind.
        """
        import onnxruntime

        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        model_input = self._session.get_inputs()[0]
        self._input = model_input.name
        self._outputs = [o.name for o in self._session.get_outputs()]
        self.kind = kind
        self.size = size or int(model_input.shape[-1])

    def predict(self, rgb: np.ndarray) -> list[detection.Detection]:
        """(H, W, 3) uint8 RGB to detections in its pixels."""
        if self.kind == "end2end":
            x, frame = letterbox(rgb, self.size)
            (rows, *_) = self._session.run(None, {self._input: x})
            return to_detections(np.asarray(rows)[0], frame)
        named = dict(
            zip(
                self._outputs,
                self._session.run(None, {self._input: stretch(rgb, self.size)}),
                strict=True,
            )
        )
        h, w = rgb.shape[:2]
        return decode_detr(
            np.asarray(named["dets"])[0], np.asarray(named["labels"])[0], w, h
        )
