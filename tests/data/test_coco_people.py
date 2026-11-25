"""Tests for minifsd.data.coco_people: the COCO person subset (D10)."""

import json
import pathlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from minifsd.data import boxes, coco_people

PERSON = 1  # COCO's own ids, as in instances_*2017.json
CHAIR = 62
TV = 72


def _coco(n_images: int = 40) -> dict[str, object]:
    """A small COCO-style file: some images with people, some without."""
    images, anns = [], []
    for i in range(1, n_images + 1):
        images.append(
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 640, "height": 480}
        )
        if i % 4 == 0:
            continue  # no people
        side = 10 if i % 5 == 0 else 80  # every fifth: only a tiny person
        anns.append(
            {
                "id": 1000 + i,
                "image_id": i,
                "category_id": PERSON,
                "bbox": [5.0, 6.0, side, side * 2],
                "area": side * side * 1.5,
                "iscrowd": 0,
            }
        )
        if i % 2 == 0:  # a chair and a TV beside the person
            for k, cat in enumerate((CHAIR, TV)):
                anns.append(
                    {
                        "id": 3000 + 10 * i + k,
                        "image_id": i,
                        "category_id": cat,
                        "bbox": [300.0, 200.0, 60.0, 80.0],
                        "area": 3000.0,
                        "iscrowd": 0,
                    }
                )
        if i % 3 == 0:  # a crowd region too
            anns.append(
                {
                    "id": 2000 + i,
                    "image_id": i,
                    "category_id": PERSON,
                    "bbox": [100.0, 100.0, 200.0, 100.0],
                    "area": 15000.0,
                    "iscrowd": 1,
                }
            )
    return {
        "images": images,
        "annotations": anns,
        "categories": [
            {"id": PERSON, "name": "person"},
            {"id": CHAIR, "name": "chair"},
            {"id": TV, "name": "tv"},
        ],
    }


def _usable_ids(coco: dict[str, object]) -> set[int]:
    anns = coco["annotations"]
    assert isinstance(anns, list)
    return {
        a["image_id"]
        for a in anns
        if a["category_id"] == PERSON
        and not a["iscrowd"]
        and a["area"] >= coco_people.MIN_PERSON_AREA
    }


def test_keeps_only_images_with_a_usable_person() -> None:
    src = _coco()
    out = coco_people.select(src, n=1000, seed=0)
    kept = {i["id"] for i in out["images"]}
    assert kept == _usable_ids(src)  # n is more than there are: all of them


def test_people_and_furniture_keep_their_labels_in_our_category_ids() -> None:
    # Option (a), 2026-11-25: COCO's furniture and clutter become obstacle,
    # so a real chair is never learned as background. Categories with no
    # match in ours (a TV) are dropped.
    out = coco_people.select(_coco(), n=1000, seed=0)
    ids = {name: i + 1 for i, name in enumerate(boxes.CATEGORIES)}
    assert [c["name"] for c in out["categories"]] == list(boxes.CATEGORIES)
    used = {a["category_id"] for a in out["annotations"]}
    assert used == {ids["person"], ids["obstacle"]}
    obstacles = [
        a for a in out["annotations"] if a["category_id"] == ids["obstacle"]
    ]
    kept = {i["id"] for i in out["images"]}
    # Every kept image with a chair has its chair labelled; none has a TV.
    assert {a["image_id"] for a in obstacles} == {i for i in kept if i % 2 == 0}
    assert len(obstacles) == len({a["image_id"] for a in obstacles})
    assert all(a["image_id"] in kept for a in out["annotations"])


def test_the_obstacle_map_names_real_coco_categories() -> None:
    # A typo in the list would silently drop a category; these are COCO
    # 2017's own names.
    coco_names = {
        "person",
        "bicycle",
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
        "tv",
    }
    assert set(coco_people.OBSTACLE_CATEGORIES) <= coco_names
    assert "tv" not in coco_people.OBSTACLE_CATEGORIES


def test_crowd_regions_are_kept_as_crowd() -> None:
    # Detectors ignore crowd regions in training and evaluation; dropping
    # them would turn real people into background.
    out = coco_people.select(_coco(), n=1000, seed=0)
    crowds = [a for a in out["annotations"] if a["iscrowd"]]
    assert crowds
    assert all(a["image_id"] % 3 == 0 for a in crowds)


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_the_subset_is_reproducible_from_its_seed(seed: int) -> None:
    src = _coco()
    a = coco_people.select(src, n=8, seed=seed)
    b = coco_people.select(src, n=8, seed=seed)
    assert a == b
    ids = [i["id"] for i in a["images"]]
    assert len(ids) == len(set(ids)) == 8
    assert ids == sorted(ids)


def test_another_seed_picks_other_images() -> None:
    src = _coco()
    pick = [
        {i["id"] for i in coco_people.select(src, n=8, seed=s)["images"]}
        for s in (0, 1)
    ]
    assert pick[0] != pick[1]


def test_the_output_is_plain_json() -> None:
    out = coco_people.select(_coco(), n=5, seed=0)
    assert json.loads(json.dumps(out)) == out


def test_download_fetches_missing_images_and_skips_present_ones(
    tmp_path: pathlib.Path,
) -> None:
    out = coco_people.select(_coco(), n=6, seed=0)
    names = [i["file_name"] for i in out["images"]]
    (tmp_path / names[0]).write_bytes(b"already here")
    fetched: list[str] = []

    def fake_fetch(url: str) -> bytes:
        fetched.append(url)
        return b"jpeg bytes"

    coco_people.download(out, "train2017", tmp_path, fetch=fake_fetch)
    assert len(fetched) == len(names) - 1
    assert all(u.startswith(coco_people.IMAGE_URL) for u in fetched)
    assert all("/train2017/" in u for u in fetched)
    assert (tmp_path / names[0]).read_bytes() == b"already here"
    assert all((tmp_path / n).read_bytes() for n in names)


def test_a_failed_download_leaves_no_half_file(tmp_path: pathlib.Path) -> None:
    out = coco_people.select(_coco(), n=2, seed=0)

    def broken(url: str) -> bytes:
        raise OSError("network down")

    with pytest.raises(OSError):
        coco_people.download(out, "train2017", tmp_path, fetch=broken, wait=0.0)
    assert list(tmp_path.iterdir()) == []


def test_a_misspelt_obstacle_category_is_caught() -> None:
    # The fixture lacks most of the real names, so the check must fail on
    # it; against the real file (main) it passes.
    with pytest.raises(ValueError, match="not COCO categories"):
        coco_people.check_obstacle_names(_coco())


def test_a_dropped_connection_is_retried(tmp_path: pathlib.Path) -> None:
    # COCO's server reset the connection once in a 5,000-image download;
    # one bad moment shouldn't end the whole run.
    out = coco_people.select(_coco(), n=3, seed=0)
    calls: dict[str, int] = {}

    def flaky(url: str) -> bytes:
        calls[url] = calls.get(url, 0) + 1
        if calls[url] < coco_people.TRIES:
            raise ConnectionResetError("reset by peer")
        return b"jpeg bytes"

    coco_people.download(out, "train2017", tmp_path, fetch=flaky, wait=0.0)
    assert all(n == coco_people.TRIES for n in calls.values())
    assert len(list(tmp_path.glob("*.jpg"))) == 3
