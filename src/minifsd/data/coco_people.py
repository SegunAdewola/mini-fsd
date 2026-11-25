"""The COCO person subset: what real people look like, for the detector (D10).

    pixi run python -m minifsd.data.coco_people --split train2017 --n 5000
        --annotations data/coco/annotations/instances_train2017.json
        --out data/coco-people/train

(one command, wrapped here to fit)

The sim has no people, and synthetic humans look worse to a detector than
real photos, so the person class is learned from COCO 2017 (decided
2026-11-25). This picks images holding at least one person big enough to
label, and renumbers categories to match the sim's COCO files
(minifsd.data.boxes.CATEGORIES).

A COCO photo labels only COCO's own categories. Kept as person-only, a
person beside a chair would teach the detector that a real chair is
background, the opposite of our "obstacle". So COCO's furniture and floor
clutter (OBSTACLE_CATEGORIES) are relabeled obstacle (option a, 2026-11-25),
which needs instances_*2017.json rather than the person-only files. Doors
and room signs have no COCO category; training must treat them as unlabeled
on COCO images, not as background.

Annotations are CC BY 4.0; the images are Flickr photos under their own
licenses.
"""

import argparse
import json
import pathlib
import time
import urllib.request
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from minifsd.data import boxes

COCO_PERSON = 1  # COCO's own category id for person
# COCO categories that are obstacles for an indoor robot: things standing on
# or near the floor. Not TVs, laptops or cups: they sit on tables or walls.
OBSTACLE_CATEGORIES = (
    "bench",
    "backpack",
    "handbag",
    "suitcase",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "refrigerator",
    "bicycle",
)
# COCO's "small" object threshold (32 x 32 px at about 640 x 480). Smaller
# people are a few pixels once the image is shrunk for the detector.
MIN_PERSON_AREA = 32 * 32
IMAGE_URL = "http://images.cocodataset.org"
WORKERS = 8  # parallel downloads
TRIES = 4  # per image; the server sometimes resets a connection
RETRY_WAIT = 2.0  # seconds before the first retry, doubling after


def select(coco: dict[str, Any], n: int, seed: int) -> dict[str, Any]:
    """Picks n images with a usable person; keeps people and obstacles.

    Args:
        coco: A COCO instances file as loaded from JSON.
        n: How many images; all usable ones if there are fewer.
        seed: Picks the same images for the same seed.

    Returns:
        A COCO dict with the chosen images (sorted by id), their person and
        OBSTACLE_CATEGORIES annotations under minifsd.data.boxes' category
        ids (crowd regions kept, flagged iscrowd = 1), and those categories.
    """
    ours = {name: i + 1 for i, name in enumerate(boxes.CATEGORIES)}
    coco_names = {c["id"]: c["name"] for c in coco["categories"]}
    to_ours = {COCO_PERSON: ours["person"]}
    for coco_id, name in coco_names.items():
        if name in OBSTACLE_CATEGORIES:
            to_ours[coco_id] = ours["obstacle"]

    usable = sorted(
        {
            a["image_id"]
            for a in coco["annotations"]
            if a["category_id"] == COCO_PERSON
            and not a["iscrowd"]
            and a["area"] >= MIN_PERSON_AREA
        }
    )
    rng = np.random.default_rng(seed)
    count = min(n, len(usable))
    chosen = set(int(i) for i in rng.choice(usable, size=count, replace=False))

    images = sorted(
        (dict(i) for i in coco["images"] if i["id"] in chosen),
        key=lambda i: i["id"],
    )
    annotations = [
        {
            "id": a["id"],
            "image_id": a["image_id"],
            "category_id": to_ours[a["category_id"]],
            "bbox": a["bbox"],
            "area": a["area"],
            "iscrowd": a["iscrowd"],
        }
        for a in coco["annotations"]
        if a["image_id"] in chosen and a["category_id"] in to_ours
    ]
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for name, i in ours.items()],
    }


def check_obstacle_names(coco: dict[str, Any]) -> None:
    """Raises ValueError if an OBSTACLE_CATEGORIES name isn't in the file.

    A misspelt name would silently drop a category, so main() checks the
    list against the real COCO file before selecting.
    """
    names = {c["name"] for c in coco["categories"]}
    missing = [n for n in OBSTACLE_CATEGORIES if n not in names]
    if missing:
        raise ValueError(f"not COCO categories: {missing}")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data: bytes = response.read()
        return data


def download(
    subset: dict[str, Any],
    split: str,
    out_dir: pathlib.Path,
    fetch: Callable[[str], bytes] = _fetch,
    wait: float = RETRY_WAIT,
) -> None:
    """Downloads the subset's images into out_dir, skipping present ones.

    Each file is written under a temporary name and renamed once complete,
    so a failed or interrupted download never leaves a half image.

    Args:
        subset: A dict from select().
        split: COCO's image folder, "train2017" or "val2017".
        out_dir: Where the images go; created if missing.
        fetch: URL to bytes; replaced in tests.
        wait: Seconds before the first retry of a failed image; doubles
            each time, up to TRIES attempts in all.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        i["file_name"]
        for i in subset["images"]
        if not (out_dir / i["file_name"]).exists()
    ]

    def get(name: str) -> None:
        url = f"{IMAGE_URL}/{split}/{name}"
        for attempt in range(TRIES):
            try:
                data = fetch(url)
                break
            except OSError:
                if attempt == TRIES - 1:
                    raise
                time.sleep(wait * 2**attempt)
        part = out_dir / f"{name}.part"
        part.write_bytes(data)
        part.rename(out_dir / name)

    with ThreadPoolExecutor(WORKERS) as pool:
        # list() re-raises the first failure here.
        list(pool.map(get, missing))


def main(argv: Sequence[str] | None = None) -> None:
    """Writes out/coco.json and downloads its images into out/."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    parser.add_argument(
        "--split", required=True, choices=("train2017", "val2017")
    )
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    with args.annotations.open() as f:
        coco = json.load(f)
    check_obstacle_names(coco)
    subset = select(coco, args.n, args.seed)
    del coco  # the full file takes a few GB in memory
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "coco.json").open("w") as f:
        json.dump(subset, f)
    person = boxes.CATEGORIES.index("person") + 1
    n_people = sum(
        a["category_id"] == person and not a["iscrowd"]
        for a in subset["annotations"]
    )
    n_obstacles = sum(a["category_id"] != person for a in subset["annotations"])
    print(
        f"{len(subset['images'])} images, {n_people} people, "
        f"{n_obstacles} obstacles; downloading"
    )
    download(subset, args.split, args.out)
    print(f"done: {args.out}/")


if __name__ == "__main__":
    main()
