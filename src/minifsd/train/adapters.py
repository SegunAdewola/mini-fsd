"""Our datasets in the layouts the trainers expect (T3).

    pixi run python -m minifsd.train.adapters yolo --out /content/yolo
        --train v9=data/v9/train coco=data/coco-people/train
        --val v9=data/v9/val coco=data/coco-people/val

(one command, wrapped here to fit; "rfdetr" in place of "yolo" for RF-DETR)

Every source is a split folder holding its images and a coco.json in our
categories (minifsd.data.boxes.CATEGORIES): v9 splits from datagen, and the
COCO people subset from coco_people. Images are symlinked, not copied, so an
adapter costs seconds on Colab. Each file name gets its source's name as a
prefix, so v9 and COCO names can't collide.

- YOLO (ultralytics): images/<split>/ and labels/<split>/ side by side, one
  txt per image with "class cx cy w h", class from 0, coordinates as
  fractions of the image. An image with nothing in it gets an empty txt,
  which teaches "no object here". Crowd regions are dropped: YOLO can't
  ignore a box, and one box around a crowd isn't a person.
- RF-DETR: one folder per split with the images and _annotations.coco.json
  (Roboflow's layout; RF-DETR names the splits train, valid and test). Crowd
  regions stay, flagged iscrowd, as COCO-style loaders skip them.
- Floor segmentation: (rgb, mask) path pairs from a v9 split.
"""

import argparse
import json
import pathlib
from collections.abc import Mapping, Sequence
from typing import Any

from minifsd.data import boxes

RFDETR_ANNOTATIONS = "_annotations.coco.json"
RFDETR_SPLITS = {"train": "train", "val": "valid", "test": "test"}
DECIMALS = 6  # in YOLO lines; a millionth of the image is far below a pixel

Sources = Mapping[str, pathlib.Path]  # prefix -> split folder


def _categories() -> list[dict[str, Any]]:
    return [{"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)]


def _load(folder: pathlib.Path) -> dict[str, Any]:
    """Reads folder/coco.json; refuses one in other categories."""
    with (folder / "coco.json").open() as f:
        coco: dict[str, Any] = json.load(f)
    found = [{"id": c["id"], "name": c["name"]} for c in coco["categories"]]
    if found != _categories():
        raise ValueError(
            f"{folder}: categories {found} are not boxes.CATEGORIES"
        )
    return coco


def _link(target: pathlib.Path, link: pathlib.Path) -> None:
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"{link} already exists")
    link.symlink_to(target.resolve())


def to_yolo_line(
    category_id: int, bbox: Sequence[float], width: int, height: int
) -> str:
    """One COCO box as a YOLO label line.

    Args:
        category_id: Our COCO category id, from 1.
        bbox: [x, y, w, h] in pixels, x, y the top-left corner. Parts
            outside the image are cut off.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        "class cx cy w h": class from 0, the rest fractions of the image.
    """
    x, y, w, h = bbox
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, width), min(y + h, height)
    numbers = (
        (x0 + x1) / 2 / width,
        (y0 + y1) / 2 / height,
        (x1 - x0) / width,
        (y1 - y0) / height,
    )
    return " ".join(
        [str(category_id - 1), *(f"{v:.{DECIMALS}f}" for v in numbers)]
    )


def from_yolo_line(
    line: str, width: int, height: int
) -> tuple[int, list[float]]:
    """The inverse of to_yolo_line: (category_id, [x, y, w, h] in pixels)."""
    cls, cx, cy, w, h = line.split()
    pw, ph = float(w) * width, float(h) * height
    x = float(cx) * width - pw / 2
    y = float(cy) * height - ph / 2
    return int(cls) + 1, [x, y, pw, ph]


def write_yolo(sources: Sources, images_dir: pathlib.Path) -> int:
    """Links images into images_dir and writes their labels beside it.

    Args:
        sources: Prefix to split folder, e.g. {"v9": .../v9/train}.
        images_dir: .../images/<split>; labels go to .../labels/<split>.

    Returns:
        The number of images written.

    Raises:
        ValueError: If images_dir's parent isn't named "images", or a
            source has other categories.
    """
    if images_dir.parent.name != "images":
        raise ValueError(f"{images_dir} must sit in a folder named images")
    labels_dir = images_dir.parent.parent / "labels" / images_dir.name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for prefix, folder in sources.items():
        coco = _load(folder)
        lines: dict[int, list[str]] = {i["id"]: [] for i in coco["images"]}
        sizes = {i["id"]: (i["width"], i["height"]) for i in coco["images"]}
        for a in coco["annotations"]:
            if a["iscrowd"]:
                continue
            width, height = sizes[a["image_id"]]
            lines[a["image_id"]].append(
                to_yolo_line(a["category_id"], a["bbox"], width, height)
            )
        for image in coco["images"]:
            name = f"{prefix}_{image['file_name']}"
            _link(folder / image["file_name"], images_dir / name)
            text = "".join(f"{line}\n" for line in lines[image["id"]])
            (labels_dir / pathlib.Path(name).with_suffix(".txt")).write_text(
                text
            )
            count += 1
    return count


def write_yolo_yaml(
    root: pathlib.Path, train: Sequence[str], val: Sequence[str]
) -> pathlib.Path:
    """Writes root/data.yaml, the file ultralytics trains from.

    It is written as JSON, which is valid YAML, so no YAML library is
    needed.

    Args:
        root: The folder holding images/ and labels/.
        train: Split names under images/ to train on.
        val: Split names under images/ to validate on.

    Returns:
        The path of data.yaml.
    """
    data = {
        "path": str(root.resolve()),
        "train": [f"images/{s}" for s in train],
        "val": [f"images/{s}" for s in val],
        "names": list(boxes.CATEGORIES),  # class k is names[k]
    }
    path = root / "data.yaml"
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def write_coco_folder(sources: Sources, out_dir: pathlib.Path) -> int:
    """Merges sources into one COCO folder: linked images plus annotations.

    Image and annotation ids are renumbered from 1, as each source numbers
    its own.

    Args:
        sources: Prefix to split folder.
        out_dir: The split folder to write, e.g. .../rfdetr/train.

    Returns:
        The number of images written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for prefix, folder in sources.items():
        coco = _load(folder)
        new_ids = {}
        for image in coco["images"]:
            name = f"{prefix}_{image['file_name']}"
            _link(folder / image["file_name"], out_dir / name)
            new_ids[image["id"]] = len(images) + 1
            images.append(
                {
                    "id": new_ids[image["id"]],
                    "file_name": name,
                    "width": image["width"],
                    "height": image["height"],
                }
            )
        for a in coco["annotations"]:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": new_ids[a["image_id"]],
                    "category_id": a["category_id"],
                    "bbox": a["bbox"],
                    "area": a["area"],
                    "iscrowd": a["iscrowd"],
                }
            )
    merged = {
        "images": images,
        "annotations": annotations,
        "categories": _categories(),
    }
    with (out_dir / RFDETR_ANNOTATIONS).open("w") as f:
        json.dump(merged, f)
    return len(images)


def floor_pairs(
    folder: pathlib.Path,
) -> list[tuple[pathlib.Path, pathlib.Path]]:
    """(rgb, floor mask) paths for every frame of a v9 split, in order.

    Raises:
        FileNotFoundError: If a frame's mask is missing.
    """
    pairs = []
    for image in _load(folder)["images"]:
        rgb = folder / image["file_name"]
        mask = folder / image["file_name"].replace("_rgb.", "_floor.")
        if not mask.exists():
            raise FileNotFoundError(f"no floor mask for {rgb}")
        pairs.append((rgb, mask))
    return pairs


def _sources(specs: Sequence[str]) -> dict[str, pathlib.Path]:
    pairs = (spec.split("=", 1) for spec in specs)
    return {prefix: pathlib.Path(folder) for prefix, folder in pairs}


def main(argv: Sequence[str] | None = None) -> None:
    """Builds a YOLO or RF-DETR dataset from v9 and COCO people splits."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("format", choices=("yolo", "rfdetr"))
    parser.add_argument("--out", required=True, type=pathlib.Path)
    for split in RFDETR_SPLITS:
        parser.add_argument(
            f"--{split}", nargs="+", default=[], metavar="NAME=FOLDER"
        )
    args = parser.parse_args(argv)

    splits = {s: _sources(getattr(args, s)) for s in RFDETR_SPLITS}
    for split, sources in splits.items():
        if not sources:
            continue
        if args.format == "yolo":
            n = write_yolo(sources, args.out / "images" / split)
        else:
            n = write_coco_folder(sources, args.out / RFDETR_SPLITS[split])
        print(f"{split}: {n} images")
    if args.format == "yolo":
        used = [s for s in ("train", "val") if splits[s]]
        print(write_yolo_yaml(args.out, used[:1], used[1:]))


if __name__ == "__main__":
    main()
