"""The real-photo labelling tool's logic, without the window or SAM (P3).

tools/label_real.py shows each photo, sends the clicks to SAM 2.1 for the
floor mask and lets boxes be dragged by hand. Everything here is the part
worth testing: the click history, window and photo coordinates (the window
shows a 4080 x 3072 photo shrunk to fit), turning a drag into a box, picking
SAM's best proposal, and saving and loading the labels.

Labels sit next to each photo, so a session can stop and resume anywhere:

    IMG_0001.jpg
    IMG_0001.floor.png    binary, the photo's size, 255 on the floor
    IMG_0001.boxes.json   {"width", "height", "boxes": [Box2D fields]}

P5 gathers the boxes into one COCO file with minifsd.data.boxes.to_coco,
the format of D8's sim boxes.
"""

import dataclasses
import json
import pathlib

import cv2
import numpy as np

from minifsd.data import boxes

EXTENSIONS = (".jpg", ".jpeg", ".png")
MASK_SUFFIX = ".floor.png"
BOXES_SUFFIX = ".boxes.json"
# Sides shorter than this, in photo pixels, are a stray click, not a box.
MIN_SIDE = 4


class Clicks:
    """The points clicked on one photo, for SAM's point prompt."""

    def __init__(self) -> None:
        """Starts with no clicks."""
        self._points: list[tuple[float, float]] = []
        self._labels: list[int] = []

    @property
    def empty(self) -> bool:
        """True with no clicks."""
        return not self._points

    @property
    def points(self) -> np.ndarray:
        """(N, 2) float (x, y) in photo pixels."""
        return np.array(self._points, np.float64).reshape(-1, 2)

    @property
    def labels(self) -> np.ndarray:
        """(N,) int: 1 for floor, 0 for not floor (SAM's convention)."""
        return np.array(self._labels, np.int64)

    def add(self, x: float, y: float, floor: bool) -> None:
        """Adds a click at (x, y) in photo pixels."""
        self._points.append((x, y))
        self._labels.append(1 if floor else 0)

    def undo(self) -> None:
        """Drops the last click, if there is one."""
        if self._points:
            self._points.pop()
            self._labels.pop()

    def reset(self) -> None:
        """Drops every click."""
        self._points.clear()
        self._labels.clear()


@dataclasses.dataclass(frozen=True)
class View:
    """How a photo is shown: shrunk by `scale` to width x height."""

    scale: float
    width: int
    height: int

    @classmethod
    def fit(
        cls,
        photo_width: int,
        photo_height: int,
        max_width: int,
        max_height: int,
    ) -> "View":
        """The largest view of the photo inside the window, never enlarged."""
        scale = min(1.0, max_width / photo_width, max_height / photo_height)
        return cls(
            scale, round(photo_width * scale), round(photo_height * scale)
        )

    def to_photo(self, u: float, v: float) -> tuple[float, float]:
        """Window pixel to photo pixel."""
        return u / self.scale, v / self.scale

    def to_window(self, x: float, y: float) -> tuple[float, float]:
        """Photo pixel to window pixel."""
        return x * self.scale, y * self.scale


def box_from_drag(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
    category: str,
) -> boxes.Box2D | None:
    """The box a mouse drag drew, in photo pixels.

    Args:
        x0: Where the drag started, x.
        y0: Where the drag started, y.
        x1: Where it ended, x; any direction from the start.
        y1: Where it ended, y.
        width: The photo's width; the box is clipped to it.
        height: The photo's height.
        category: One of boxes.CATEGORIES.

    Returns:
        The box, or None if it is under MIN_SIDE on a side once clipped.
    """
    left = int(np.clip(round(min(x0, x1)), 0, width))
    right = int(np.clip(round(max(x0, x1)), 0, width))
    top = int(np.clip(round(min(y0, y1)), 0, height))
    bottom = int(np.clip(round(max(y0, y1)), 0, height))
    w, h = right - left, bottom - top
    if w < MIN_SIDE or h < MIN_SIDE:
        return None
    # Hand-drawn: no pixel count, so the box's area stands in for it.
    return boxes.Box2D(category, f"{category}_drawn", left, top, w, h, w * h)


def best_mask(logits: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The proposal SAM scores highest, as a bool mask.

    Args:
        logits: (3, H, W) mask logits, SAM's three proposals.
        scores: (3,) SAM's predicted IoU for each.

    Returns:
        (H, W) bool, True where that proposal's logit is above 0.
    """
    return np.asarray(logits[int(np.argmax(scores))] > 0.0)


@dataclasses.dataclass(frozen=True)
class Labels:
    """One photo's labels."""

    floor: np.ndarray  # (H, W) bool
    boxes: list[boxes.Box2D]


def mask_path(photo: pathlib.Path) -> pathlib.Path:
    """Where a photo's floor mask is saved."""
    return photo.with_name(photo.stem + MASK_SUFFIX)


def boxes_path(photo: pathlib.Path) -> pathlib.Path:
    """Where a photo's boxes are saved."""
    return photo.with_name(photo.stem + BOXES_SUFFIX)


def _size(photo: pathlib.Path) -> tuple[int, int]:
    # Read as everything else reads photos: cv2.imread's default turns the
    # image by its EXIF orientation, IMREAD_UNCHANGED doesn't, and a mask
    # must match the turned image.
    image = cv2.imread(str(photo))
    if image is None:
        raise FileNotFoundError(f"can't read {photo}")
    return image.shape[1], image.shape[0]


def save(photo: pathlib.Path, labels: Labels) -> None:
    """Writes the mask and boxes next to the photo.

    Raises:
        ValueError: If the mask isn't the photo's size.
    """
    width, height = _size(photo)
    if labels.floor.shape != (height, width):
        raise ValueError(
            f"mask is {labels.floor.shape}, photo is {(height, width)}"
        )
    mask = np.where(labels.floor, 255, 0).astype(np.uint8)
    cv2.imwrite(str(mask_path(photo)), mask)
    record = {
        "width": width,
        "height": height,
        "boxes": [dataclasses.asdict(b) for b in labels.boxes],
    }
    boxes_path(photo).write_text(json.dumps(record, indent=1) + "\n")


def load(photo: pathlib.Path) -> Labels | None:
    """The photo's saved labels, or None if it has none yet."""
    if not (mask_path(photo).exists() and boxes_path(photo).exists()):
        return None
    mask = cv2.imread(str(mask_path(photo)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"can't read {mask_path(photo)}")
    record = json.loads(boxes_path(photo).read_text())
    found = [boxes.Box2D(**b) for b in record["boxes"]]
    return Labels(np.asarray(mask > 127), found)


def photos(folder: pathlib.Path) -> list[pathlib.Path]:
    """The folder's photos in name order, leaving out label files."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.suffix.lower() in EXTENSIONS and not p.name.endswith(MASK_SUFFIX)
    )


def unlabelled(folder: pathlib.Path) -> list[pathlib.Path]:
    """The folder's photos that have no saved labels yet."""
    return [p for p in photos(folder) if load(p) is None]
