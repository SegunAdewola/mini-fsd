"""Tests for minifsd.data.real_labels: the labelling tool's logic (P3)."""

import pathlib

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.data import boxes, real_labels


def test_clicks_add_undo_and_reset() -> None:
    clicks = real_labels.Clicks()
    assert clicks.empty
    clicks.add(10.0, 20.0, floor=True)
    clicks.add(30.0, 40.0, floor=False)
    np.testing.assert_array_equal(clicks.points, [[10, 20], [30, 40]])
    np.testing.assert_array_equal(clicks.labels, [1, 0])  # SAM: 1 is "this"
    clicks.undo()
    np.testing.assert_array_equal(clicks.points, [[10, 20]])
    clicks.reset()
    assert clicks.empty
    clicks.undo()  # undo with nothing to undo is harmless
    assert clicks.empty


def test_a_big_photo_is_shrunk_to_fit_the_window_and_never_enlarged() -> None:
    view = real_labels.View.fit(4080, 3072, max_width=1600, max_height=1000)
    assert view.width <= 1600 and view.height <= 1000
    assert view.width == pytest.approx(4080 * view.scale, abs=1)
    small = real_labels.View.fit(320, 240, max_width=1600, max_height=1000)
    assert small.scale == 1.0


@given(st.floats(0, 4079), st.floats(0, 3071))
def test_window_and_photo_coordinates_round_trip(x: float, y: float) -> None:
    view = real_labels.View.fit(4080, 3072, max_width=1600, max_height=1000)
    u, v = view.to_window(x, y)
    assert view.to_photo(u, v) == pytest.approx((x, y), abs=1e-6)


@given(
    st.floats(-50, 450),
    st.floats(-50, 350),
    st.floats(-50, 450),
    st.floats(-50, 350),
)
def test_a_drag_in_any_direction_gives_a_box_inside_the_photo(
    x0: float, y0: float, x1: float, y1: float
) -> None:
    box = real_labels.box_from_drag(x0, y0, x1, y1, 400, 300, "door")
    if box is None:  # too small once clipped: a stray click, not a box
        return
    assert box.category == "door"
    assert box.x >= 0 and box.x + box.w <= 400
    assert box.y >= 0 and box.y + box.h <= 300
    assert box.w >= real_labels.MIN_SIDE and box.h >= real_labels.MIN_SIDE


def test_a_click_without_a_drag_is_no_box() -> None:
    assert real_labels.box_from_drag(50, 50, 51, 52, 400, 300, "door") is None


def test_the_best_scoring_proposal_is_kept() -> None:
    logits = np.full((3, 4, 5), -1.0, np.float32)
    logits[1, :2] = 2.0  # proposal 1: the top half
    mask = real_labels.best_mask(logits, np.array([0.2, 0.9, 0.5]))
    assert mask.dtype == bool and mask.shape == (4, 5)
    assert mask[:2].all() and not mask[2:].any()


def _photo(tmp_path: pathlib.Path, name: str = "IMG_0001.jpg") -> pathlib.Path:
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((30, 40, 3), np.uint8))
    return path


def test_labels_round_trip_and_the_mask_is_binary_and_full_size(
    tmp_path: pathlib.Path,
) -> None:
    photo = _photo(tmp_path)
    floor = np.zeros((30, 40), bool)
    floor[20:] = True
    found = [boxes.Box2D("door", "door_0", 5, 2, 10, 20, 200)]
    real_labels.save(photo, real_labels.Labels(floor, found))
    stored = cv2.imread(str(real_labels.mask_path(photo)), cv2.IMREAD_UNCHANGED)
    assert stored is not None
    assert stored.shape == (30, 40)
    assert set(np.unique(stored)) <= {0, 255}
    back = real_labels.load(photo)
    assert back is not None
    np.testing.assert_array_equal(back.floor, floor)
    assert back.boxes == found


def test_a_mask_of_the_wrong_size_is_refused(tmp_path: pathlib.Path) -> None:
    photo = _photo(tmp_path)
    with pytest.raises(ValueError):
        real_labels.save(
            photo, real_labels.Labels(np.zeros((10, 10), bool), [])
        )


def test_finished_photos_are_skipped_on_resume(tmp_path: pathlib.Path) -> None:
    done = _photo(tmp_path, "IMG_0001.jpg")
    todo = _photo(tmp_path, "IMG_0002.jpg")
    real_labels.save(done, real_labels.Labels(np.zeros((30, 40), bool), []))
    # Label files never count as photos themselves.
    assert real_labels.photos(tmp_path) == [done, todo]
    assert real_labels.unlabelled(tmp_path) == [todo]
