"""2D object boxes from a label image, and the COCO file that holds them (D8).

The renderer says which geom every pixel shows. Each geom that is part of a
labelled object (a box of clutter, a door, a room sign) maps to a category
and an instance; an instance's box is the tight extent of its visible
pixels. That is a modal box, the visible part only, as COCO draws them.

The person category is listed so that the category ids match the COCO
person subset (D10); sim frames have no people.
"""

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

CATEGORIES = ("obstacle", "door", "room_sign", "person")
# Under this many visible pixels (of 320 x 240) an object is too small or
# too hidden to label: a detector can't be asked to find it.
MIN_PIXELS = 40


@dataclasses.dataclass(frozen=True)
class Box2D:
    """One object's visible extent in one frame, in pixels.

    Attributes:
        category: One of CATEGORIES.
        instance: Which object, e.g. "door_101", unique within a frame.
        x: Left column.
        y: Top row.
        w: Width; the box covers columns x to x + w - 1.
        h: Height.
        pixels: How many pixels of the object are visible.
    """

    category: str
    instance: str
    x: int
    y: int
    w: int
    h: int
    pixels: int


def boxes_from_ids(
    ids: np.ndarray,
    objects: Mapping[int, tuple[str, str]],
    min_pixels: int = MIN_PIXELS,
) -> list[Box2D]:
    """One box per object with at least min_pixels visible.

    Args:
        ids: (H, W) int, the geom id each pixel shows (-1 for none).
        objects: Geom id to (category, instance). Several geoms can share an
            instance (a door's frame and panel); geoms not listed are not
            objects (floor, walls).
        min_pixels: Fewest visible pixels for an object to get a box.

    Returns:
        The boxes, in order of instance name.

    Raises:
        ValueError: For a category not in CATEGORIES.
    """
    by_instance: dict[str, tuple[str, list[int]]] = {}
    for geom, (category, instance) in objects.items():
        if category not in CATEGORIES:
            raise ValueError(
                f"category must be one of {CATEGORIES}, got {category!r}"
            )
        by_instance.setdefault(instance, (category, []))[1].append(geom)

    found = []
    for instance in sorted(by_instance):
        category, geoms = by_instance[instance]
        mask = np.isin(ids, geoms)
        pixels = int(mask.sum())
        if pixels < min_pixels:
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        found.append(
            Box2D(
                category=category,
                instance=instance,
                x=int(cols[0]),
                y=int(rows[0]),
                w=int(cols[-1] - cols[0] + 1),
                h=int(rows[-1] - rows[0] + 1),
                pixels=pixels,
            )
        )
    return found


def to_coco(
    images: Sequence[tuple[str, int, int, Sequence[Box2D]]],
) -> dict[str, object]:
    """A COCO detection file for a set of images.

    Args:
        images: (file name, width, height, boxes) per image.

    Returns:
        The COCO dict, ready for json.dump. Category ids are 1 to 4 in
        CATEGORIES order, image ids 1 to N in the given order. "area" is the
        visible pixel count, not the box's area.
    """
    category_id = {name: i + 1 for i, name in enumerate(CATEGORIES)}
    coco_images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for image_id, (name, width, height, found) in enumerate(images, start=1):
        coco_images.append(
            {
                "id": image_id,
                "file_name": name,
                "width": width,
                "height": height,
            }
        )
        for b in found:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id[b.category],
                    "bbox": [b.x, b.y, b.w, b.h],
                    "area": b.pixels,
                    "iscrowd": 0,
                }
            )
    return {
        "images": coco_images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name} for name, i in category_id.items()
        ],
    }
