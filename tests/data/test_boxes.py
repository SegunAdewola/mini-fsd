"""Tests for minifsd.data.boxes: 2D boxes from a label image, and COCO."""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.data import boxes

# Geom ids 3 and 4 are one door (frame and panel), 7 is a sign, 9 a box.
OBJECTS = {
    3: ("door", "door_101"),
    4: ("door", "door_101"),
    7: ("room_sign", "sign_101"),
    9: ("obstacle", "clutter_0"),
}


def test_a_box_is_the_tight_extent_of_the_visible_pixels() -> None:
    ids = np.full((20, 30), -1)
    ids[5:9, 10:16] = 7  # rows 5 to 8, cols 10 to 15
    (b,) = boxes.boxes_from_ids(ids, OBJECTS, min_pixels=1)
    assert (b.category, b.instance) == ("room_sign", "sign_101")
    assert (b.x, b.y, b.w, b.h) == (10, 5, 6, 4)
    assert b.pixels == 24


def test_geoms_of_one_object_make_one_box() -> None:
    ids = np.full((20, 30), -1)
    ids[2:18, 4:6] = 3  # one jamb
    ids[2:18, 12:14] = 3  # the other
    ids[2:18, 6:12] = 4  # the panel between them
    (b,) = boxes.boxes_from_ids(ids, OBJECTS, min_pixels=1)
    assert (b.x, b.y, b.w, b.h) == (4, 2, 10, 16)


def test_small_and_hidden_objects_get_no_box() -> None:
    ids = np.full((20, 30), -1)
    ids[0:2, 0:2] = 9  # 4 pixels of the obstacle
    assert boxes.boxes_from_ids(ids, OBJECTS, min_pixels=5) == []
    # The sign isn't in the image at all.
    assert all(
        b.instance != "sign_101" for b in boxes.boxes_from_ids(ids, OBJECTS, 1)
    )


def test_ids_that_are_not_objects_are_ignored() -> None:
    ids = np.zeros((10, 10), int)  # geom 0, the floor, everywhere
    assert boxes.boxes_from_ids(ids, OBJECTS, min_pixels=1) == []


@settings(max_examples=50, deadline=None)
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_boxes_lie_inside_the_image_and_hold_their_pixels(seed: int) -> None:
    rng = np.random.default_rng(seed)
    ids = rng.choice([-1, 0, 3, 4, 7, 9], size=(24, 32))
    found = boxes.boxes_from_ids(ids, OBJECTS, min_pixels=1)
    for b in found:
        assert b.x >= 0 and b.x + b.w <= 32
        assert b.y >= 0 and b.y + b.h <= 24
        mine = [g for g, (_, inst) in OBJECTS.items() if inst == b.instance]
        assert b.pixels == int(np.isin(ids, mine).sum())
        assert b.pixels <= b.w * b.h
    assert len({b.instance for b in found}) == len(found)


def test_rejects_an_unknown_category() -> None:
    with pytest.raises(ValueError):
        boxes.boxes_from_ids(
            np.zeros((4, 4), int), {0: ("chair", "c0")}, min_pixels=1
        )


def test_coco_file_has_the_categories_in_a_fixed_order() -> None:
    sign = boxes.Box2D("room_sign", "sign_101", 10, 5, 6, 4, 24)
    door = boxes.Box2D("door", "door_101", 4, 2, 10, 16, 150)
    coco = boxes.to_coco(
        [
            ("a_rgb.png", 320, 240, [sign, door]),
            ("b_rgb.png", 320, 240, []),
        ]
    )
    coco = json.loads(json.dumps(coco))  # it must be plain JSON
    assert [c["name"] for c in coco["categories"]] == list(boxes.CATEGORIES)
    assert [c["id"] for c in coco["categories"]] == [1, 2, 3, 4]
    assert [i["file_name"] for i in coco["images"]] == [
        "a_rgb.png",
        "b_rgb.png",
    ]
    assert len({i["id"] for i in coco["images"]}) == 2
    by_name = {c["name"]: c["id"] for c in coco["categories"]}
    (a_id,) = (i["id"] for i in coco["images"] if i["file_name"] == "a_rgb.png")
    anns = coco["annotations"]
    assert len(anns) == 2 and len({a["id"] for a in anns}) == 2
    first = next(a for a in anns if a["category_id"] == by_name["room_sign"])
    assert first["image_id"] == a_id
    assert first["bbox"] == [10, 5, 6, 4]
    assert first["area"] == 24 and first["iscrowd"] == 0
