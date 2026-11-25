"""Detection accuracy: average precision at IoU 0.5, per class and mean (T1).

The design reports detection mAP@0.5 for doors, signs, obstacles and people.
This is the PASCAL VOC way of computing it, written here rather than adding
pycocotools:

1. Per category, sort every image's detections by score, highest first.
2. Each detection matches the unmatched ground-truth box of its category in
   its image that it overlaps most, if that IoU is at least 0.5: a hit.
   Otherwise it is a false alarm. So a second detection of one object is a
   false alarm.
3. Precision and recall after each detection trace a curve; AP is the area
   under its upper envelope (all-point interpolation, VOC 2010 on).

Crowd regions (COCO's iscrowd) follow COCO's rule: they aren't targets, and
a detection that lies mostly inside one (intersection over the detection's
own area at least 0.5) is ignored, neither hit nor false alarm.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np

IOU_THRESHOLD = 0.5


@dataclasses.dataclass(frozen=True)
class Truth:
    """A ground-truth box: x, y is the top-left corner, in pixels."""

    category: str
    x: float
    y: float
    w: float
    h: float
    crowd: bool = False


@dataclasses.dataclass(frozen=True)
class Detection:
    """A predicted box with the model's confidence."""

    category: str
    x: float
    y: float
    w: float
    h: float
    score: float


Box = Truth | Detection


def _intersection(a: Box, b: Box) -> float:
    w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return max(w, 0.0) * max(h, 0.0)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    inter = _intersection(a, b)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def average_precision(
    found: Sequence[Sequence[Detection]],
    truth: Sequence[Sequence[Truth]],
    category: str,
) -> float:
    """AP at IoU 0.5 for one category over a set of images.

    Args:
        found: Per image, the model's detections (any categories).
        truth: Per image, the ground-truth boxes, in the same order.
        category: The category to score.

    Returns:
        AP in [0, 1].

    Raises:
        ValueError: If the category has no (non-crowd) ground truth, so AP
            is undefined.
    """
    targets = [
        [t for t in image if t.category == category and not t.crowd]
        for image in truth
    ]
    crowds = [
        [t for t in image if t.category == category and t.crowd]
        for image in truth
    ]
    n_targets = sum(len(t) for t in targets)
    if n_targets == 0:
        raise ValueError(f"no ground truth for {category!r}")

    ranked = sorted(
        (
            (d.score, i, d)
            for i, image in enumerate(found)
            for d in image
            if d.category == category
        ),
        key=lambda item: -item[0],
    )
    taken = [np.zeros(len(t), dtype=bool) for t in targets]
    hits = []
    for _, i, d in ranked:
        overlaps = [iou(d, t) for t in targets[i]]
        free = [k for k, o in enumerate(overlaps) if not taken[i][k]]
        best = max(free, key=lambda k: overlaps[k], default=None)
        if best is not None and overlaps[best] >= IOU_THRESHOLD:
            taken[i][best] = True
            hits.append(True)
        elif any(
            _intersection(d, c) / (d.w * d.h) >= IOU_THRESHOLD
            for c in crowds[i]
            if d.w * d.h > 0
        ):
            continue  # inside a crowd region: ignored
        else:
            hits.append(False)

    if not hits:
        return 0.0
    tp = np.cumsum(hits)
    fp = np.cumsum(np.logical_not(hits))
    recall = tp / n_targets
    precision = tp / (tp + fp)
    # Upper envelope: precision at recall r is the best at any recall >= r.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.diff(np.concatenate([[0.0], recall]))
    return float((steps * envelope).sum())


def mean_average_precision(
    found: Sequence[Sequence[Detection]],
    truth: Sequence[Sequence[Truth]],
) -> dict[str, float]:
    """AP per category with ground truth, and their mean.

    Args:
        found: Per image, the model's detections.
        truth: Per image, the ground-truth boxes.

    Returns:
        {"ap50_<category>": AP, ..., "map50": mean over those categories}.
        A category with no ground truth is left out, not counted as 0.

    Raises:
        ValueError: If there is no ground truth at all, or the lists differ
            in length.
    """
    if len(found) != len(truth):
        raise ValueError(
            f"{len(found)} detection lists for {len(truth)} images"
        )
    categories = sorted(
        {t.category for image in truth for t in image if not t.crowd}
    )
    if not categories:
        raise ValueError("no ground truth to score against")
    scores = {
        f"ap50_{c}": average_precision(found, truth, c) for c in categories
    }
    scores["map50"] = float(np.mean(list(scores.values())))
    return scores
