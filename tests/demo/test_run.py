"""Tests for minifsd.demo.run: a folder of photos in, overlays out."""

import pathlib

import cv2
import numpy as np
import pytest

from minifsd.demo import run


def _write(path: pathlib.Path, h: int, w: int) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (h, w, 3), np.uint8)
    image[h // 2 :] = (90, 90, 90)  # a flat grey "floor" in the lower half
    assert cv2.imwrite(str(path), image)


def test_two_images_give_two_overlays_of_the_same_size(
    tmp_path: pathlib.Path,
) -> None:
    photos, out = tmp_path / "photos", tmp_path / "out"
    photos.mkdir()
    _write(photos / "a.png", 240, 320)
    # Bigger than the working size, like a phone photo, and an upper-case
    # extension, as phones write them.
    _write(photos / "b.JPG", 960, 1280)

    run.main(["--input", str(photos), "--output", str(out)])

    for name, shape in (("a", (240, 320, 3)), ("b", (960, 1280, 3))):
        result = cv2.imread(str(out / f"{name}_overlay.png"))
        assert result is not None, f"no overlay for {name}"
        assert result.shape == shape


def test_finds_only_images_in_name_order(tmp_path: pathlib.Path) -> None:
    for name in ("c.jpeg", "a.png", "notes.txt", "b.jpg"):
        (tmp_path / name).touch()
    found = run.find_images(tmp_path)
    assert [p.name for p in found] == ["a.png", "b.jpg", "c.jpeg"]


def test_an_empty_folder_is_an_error(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit):
        run.main(["--input", str(tmp_path), "--output", str(tmp_path / "o")])


def test_the_floor_baseline_sees_at_most_the_working_size() -> None:
    seen: list[tuple[int, ...]] = []

    def fake_floor(rgb: np.ndarray) -> np.ndarray:
        seen.append(rgb.shape)
        return np.ones(rgb.shape[:2], bool)

    out = run.process(np.zeros((960, 1280, 3), np.uint8), floor=fake_floor)
    assert max(seen[0][:2]) == run.WORKING_SIZE
    assert seen[0][1] / seen[0][0] == pytest.approx(1280 / 960, rel=0.01)
    assert out.shape == (960, 1280, 3)
