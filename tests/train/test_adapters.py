"""Tests for minifsd.train.adapters: our COCO files to each trainer (T3)."""

import json
import pathlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.data import boxes
from minifsd.train import adapters

WIDTH, HEIGHT = 320, 240


def _box_in_image(draw: st.DrawFn) -> tuple[float, float, float, float]:
    x = draw(st.floats(0, WIDTH - 1))
    y = draw(st.floats(0, HEIGHT - 1))
    w = draw(st.floats(1, WIDTH - x))
    h = draw(st.floats(1, HEIGHT - y))
    return x, y, w, h


BOXES = st.composite(_box_in_image)()


def _dataset(
    folder: pathlib.Path,
    images: list[tuple[str, list[tuple[int, list[float], int]]]],
    suffix: str = ".png",
) -> pathlib.Path:
    """A split folder with coco.json and tiny stand-in image files.

    Args:
        folder: Where to write it.
        images: Per image, a stem and its (category_id, bbox, iscrowd).
        suffix: The image file extension.
    """
    folder.mkdir(parents=True)
    coco: dict[str, list[dict[str, object]]] = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)
        ],
    }
    for image_id, (stem, found) in enumerate(images, start=1):
        name = f"{stem}{suffix}"
        (folder / name).write_bytes(stem.encode())
        coco["images"].append(
            {
                "id": image_id,
                "file_name": name,
                "width": WIDTH,
                "height": HEIGHT,
            }
        )
        for category_id, bbox, crowd in found:
            coco["annotations"].append(
                {
                    "id": len(coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": crowd,
                }
            )
    (folder / "coco.json").write_text(json.dumps(coco))
    return folder


@given(BOXES, st.integers(1, len(boxes.CATEGORIES)))
def test_a_box_survives_the_round_trip_through_a_yolo_line(
    bbox: tuple[float, float, float, float], category_id: int
) -> None:
    line = adapters.to_yolo_line(category_id, list(bbox), WIDTH, HEIGHT)
    back_id, back = adapters.from_yolo_line(line, WIDTH, HEIGHT)
    assert back_id == category_id
    # Six decimals of a fraction of 320 px: well under a thousandth of a px.
    assert back == pytest.approx(list(bbox), abs=1e-3)


@given(BOXES)
def test_yolo_lines_use_zero_based_classes_and_unit_coordinates(
    bbox: tuple[float, float, float, float],
) -> None:
    line = adapters.to_yolo_line(1, list(bbox), WIDTH, HEIGHT)
    cls, *numbers = line.split()
    assert cls == "0"  # obstacle, COCO id 1
    assert all(0.0 <= float(v) <= 1.0 for v in numbers)


def test_a_box_poking_out_of_the_image_is_clipped() -> None:
    line = adapters.to_yolo_line(2, [-5, 10, 30, 300], WIDTH, HEIGHT)
    _, back = adapters.from_yolo_line(line, WIDTH, HEIGHT)
    assert back == pytest.approx([0, 10, 25, HEIGHT - 10], abs=1e-3)


def test_yolo_folder_keeps_every_image_and_its_labels(
    tmp_path: pathlib.Path,
) -> None:
    sim = _dataset(
        tmp_path / "v9" / "train",
        [
            ("frame_a_rgb", [(2, [10, 20, 30, 40], 0), (3, [50, 5, 8, 6], 0)]),
            ("frame_b_rgb", []),  # nothing in view: a background image
        ],
    )
    people = _dataset(
        tmp_path / "coco" / "train",
        [("0001", [(4, [1.5, 2.5, 60, 100], 0), (1, [100, 50, 40, 40], 0)])],
        suffix=".jpg",
    )
    out = tmp_path / "yolo"
    count = adapters.write_yolo(
        {"v9": sim, "coco": people}, out / "images" / "train"
    )
    assert count == 3
    images = sorted(p.name for p in (out / "images" / "train").iterdir())
    assert images == [
        "coco_0001.jpg",
        "v9_frame_a_rgb.png",
        "v9_frame_b_rgb.png",
    ]
    assert (out / "images" / "train" / "v9_frame_a_rgb.png").read_bytes() == (
        b"frame_a_rgb"
    )
    labels = out / "labels" / "train"
    a = (labels / "v9_frame_a_rgb.txt").read_text().splitlines()
    assert [adapters.from_yolo_line(x, WIDTH, HEIGHT) for x in a] == [
        (2, pytest.approx([10, 20, 30, 40], abs=1e-3)),
        (3, pytest.approx([50, 5, 8, 6], abs=1e-3)),
    ]
    assert (labels / "v9_frame_b_rgb.txt").read_text() == ""
    people_lines = (labels / "coco_0001.txt").read_text().splitlines()
    assert [line.split()[0] for line in people_lines] == ["3", "0"]


def test_yolo_drops_crowd_regions(tmp_path: pathlib.Path) -> None:
    # YOLO has no "ignore" label: a crowd box would teach one person-shaped
    # box around a whole crowd, so it is left out.
    people = _dataset(
        tmp_path / "coco",
        [("0001", [(4, [0, 0, 50, 90], 0), (4, [100, 0, 200, 200], 1)])],
    )
    adapters.write_yolo({"coco": people}, tmp_path / "y" / "images" / "val")
    lines = (tmp_path / "y" / "labels" / "val" / "coco_0001.txt").read_text()
    assert len(lines.splitlines()) == 1


def test_yolo_needs_an_images_folder(tmp_path: pathlib.Path) -> None:
    # Ultralytics finds labels by swapping /images/ for /labels/ in a path.
    sim = _dataset(tmp_path / "v9", [("a", [])])
    with pytest.raises(ValueError, match="images"):
        adapters.write_yolo({"v9": sim}, tmp_path / "out" / "train")


def test_data_yaml_names_classes_in_category_order(
    tmp_path: pathlib.Path,
) -> None:
    path = adapters.write_yolo_yaml(tmp_path, ["train"], ["val"])
    # JSON is valid YAML, so no YAML library is needed to write it.
    data = json.loads(path.read_text())
    assert data["path"] == str(tmp_path.resolve())
    assert data["train"] == ["images/train"]
    assert data["val"] == ["images/val"]
    # A list, as JSON keys can't be ints; class k is names[k].
    assert data["names"] == list(boxes.CATEGORIES)


def test_coco_merge_renumbers_ids_and_keeps_every_box(
    tmp_path: pathlib.Path,
) -> None:
    sim = _dataset(
        tmp_path / "v9",
        [("a", [(2, [10, 20, 30, 40], 0)]), ("b", [(3, [1, 2, 3, 4], 0)])],
    )
    people = _dataset(
        tmp_path / "coco",
        [("0001", [(4, [5, 6, 7, 8], 0), (4, [0, 0, 90, 90], 1)])],
        suffix=".jpg",
    )
    out = tmp_path / "rfdetr" / "train"
    count = adapters.write_coco_folder({"v9": sim, "coco": people}, out)
    assert count == 3
    merged = json.loads((out / adapters.RFDETR_ANNOTATIONS).read_text())
    image_ids = [i["id"] for i in merged["images"]]
    assert len(set(image_ids)) == len(image_ids)
    ann_ids = [a["id"] for a in merged["annotations"]]
    assert len(set(ann_ids)) == len(ann_ids)
    by_name = {i["file_name"]: i["id"] for i in merged["images"]}
    assert set(by_name) == {"v9_a.png", "v9_b.png", "coco_0001.jpg"}
    for name in by_name:
        assert (out / name).exists()
    found = {
        (a["image_id"], a["category_id"], tuple(a["bbox"]), a["iscrowd"])
        for a in merged["annotations"]
    }
    assert found == {
        (by_name["v9_a.png"], 2, (10, 20, 30, 40), 0),
        (by_name["v9_b.png"], 3, (1, 2, 3, 4), 0),
        (by_name["coco_0001.jpg"], 4, (5, 6, 7, 8), 0),
        (by_name["coco_0001.jpg"], 4, (0, 0, 90, 90), 1),  # kept: DETR ignores
    }
    assert merged["categories"] == [
        {"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)
    ]


def test_a_dataset_with_other_categories_is_refused(
    tmp_path: pathlib.Path,
) -> None:
    folder = _dataset(tmp_path / "v9", [("a", [])])
    coco = json.loads((folder / "coco.json").read_text())
    coco["categories"][0]["name"] = "chair"
    (folder / "coco.json").write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="categories"):
        adapters.write_coco_folder({"v9": folder}, tmp_path / "out")
    with pytest.raises(ValueError, match="categories"):
        adapters.write_yolo({"v9": folder}, tmp_path / "y" / "images" / "t")


def test_floor_pairs_match_each_frame_to_its_mask(
    tmp_path: pathlib.Path,
) -> None:
    folder = _dataset(tmp_path / "v9", [("p_0000_rgb", []), ("p_0001_rgb", [])])
    for stem in ("p_0000", "p_0001"):
        (folder / f"{stem}_floor.png").write_bytes(b"mask")
    pairs = adapters.floor_pairs(folder)
    assert [(r.name, m.name) for r, m in pairs] == [
        ("p_0000_rgb.png", "p_0000_floor.png"),
        ("p_0001_rgb.png", "p_0001_floor.png"),
    ]


def test_floor_pairs_refuse_a_frame_without_a_mask(
    tmp_path: pathlib.Path,
) -> None:
    folder = _dataset(tmp_path / "v9", [("p_0000_rgb", [])])
    with pytest.raises(FileNotFoundError):
        adapters.floor_pairs(folder)
