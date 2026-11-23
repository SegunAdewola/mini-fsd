"""Tests for minifsd.data.nyuv2.

The tests write a tiny fake .mat file in the same layout as the real one, so
they never need the 3 GB download. One test reads the real file and is skipped
when it is missing (always, in CI).

The fake images are 2 high by 4 wide with 3 channels. All three sizes differ,
and every pixel holds a different value, so any wrong axis order shows up as a
wrong shape or wrong values. It never passes by accident.
"""

import pathlib
import tempfile

import h5py
import numpy as np
import pytest
import scipy.io
from hypothesis import given
from hypothesis import strategies as st

from minifsd.data import nyuv2

H, W = 2, 4
REAL_MAT = pathlib.Path("data/nyuv2/nyu_depth_v2_labeled.mat")
REAL_SPLITS = pathlib.Path("data/nyuv2/splits.mat")

# Ground truth for the fake file, in the layout we want out: (H, W, ...).
RGB = np.arange(2 * H * W * 3, dtype=np.uint8).reshape(2, H, W, 3)
DEPTH = (np.arange(2 * H * W, dtype=np.float32).reshape(2, H, W) + 1) / 4
CLASS_NAMES = ["wall", "floor", "chair"]  # so floor's label value is 2
LABELS = np.array(
    [
        [[2, 2, 1, 0], [2, 3, 0, 1]],
        [[0, 0, 0, 0], [2, 2, 2, 2]],
    ],
    dtype=np.uint16,
)


def _write_fake_mat(path: pathlib.Path, class_names: list[str]) -> None:
    """Writes a .mat (HDF5) file laid out like nyu_depth_v2_labeled.mat."""
    with h5py.File(path, "w") as f:
        # MATLAB layout: axes reversed. (n, H, W, C) becomes (n, C, W, H).
        f["images"] = RGB.transpose(0, 3, 2, 1)
        f["depths"] = DEPTH.transpose(0, 2, 1)
        f["labels"] = LABELS.transpose(0, 2, 1)
        # Class names: each stored as character codes, referenced from a
        # (1, n_classes) array, which is how MATLAB writes a cell array.
        refs = f.create_dataset(
            "names", shape=(1, len(class_names)), dtype=h5py.ref_dtype
        )
        for i, name in enumerate(class_names):
            codes = np.array([[ord(c)] for c in name], dtype=np.uint16)
            refs[0, i] = f.create_dataset(f"#refs#/{i}", data=codes).ref


@pytest.fixture
def fake_mat(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "fake.mat"
    _write_fake_mat(path, CLASS_NAMES)
    return path


def _write_splits(
    path: pathlib.Path, train: list[int], test: list[int]
) -> None:
    """Writes splits.mat with 1-based indices as columns, like the real one."""
    scipy.io.savemat(
        str(path),
        {
            "trainNdxs": np.array(train, dtype=float).reshape(-1, 1),
            "testNdxs": np.array(test, dtype=float).reshape(-1, 1),
        },
    )


# --- Hole 1: the transpose --------------------------------------------------


def test_rgb_has_height_width_channel_layout(fake_mat: pathlib.Path) -> None:
    with nyuv2.NyuV2(fake_mat) as data:
        sample = data[0]
    assert sample.rgb.shape == (H, W, 3)
    assert sample.rgb.dtype == np.uint8


def test_rgb_pixels_match_the_original(fake_mat: pathlib.Path) -> None:
    with nyuv2.NyuV2(fake_mat) as data:
        for i in range(2):
            np.testing.assert_array_equal(data[i].rgb, RGB[i])


# --- Depth and labels -------------------------------------------------------


def test_depth_comes_through_unchanged(fake_mat: pathlib.Path) -> None:
    with nyuv2.NyuV2(fake_mat) as data:
        sample = data[1]
    assert sample.depth.dtype == np.float32
    np.testing.assert_array_equal(sample.depth, DEPTH[1])


def test_floor_mask_marks_floor_pixels(fake_mat: pathlib.Path) -> None:
    with nyuv2.NyuV2(fake_mat) as data:
        for i in range(2):
            np.testing.assert_array_equal(data[i].floor, LABELS[i] == 2)


def test_label_zero_is_marked_unlabeled(fake_mat: pathlib.Path) -> None:
    with nyuv2.NyuV2(fake_mat) as data:
        for i in range(2):
            np.testing.assert_array_equal(data[i].labeled, LABELS[i] != 0)


@pytest.mark.parametrize(
    "class_names",
    [["floor", "wall", "chair"], ["wall", "chair", "floor"]],
)
def test_floor_is_found_by_name_wherever_it_is(
    tmp_path: pathlib.Path, class_names: list[str]
) -> None:
    path = tmp_path / "fake.mat"
    _write_fake_mat(path, class_names)
    floor_value = class_names.index("floor") + 1
    with nyuv2.NyuV2(path) as data:
        np.testing.assert_array_equal(data[0].floor, LABELS[0] == floor_value)


def test_missing_floor_class_raises(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "fake.mat"
    _write_fake_mat(path, ["wall", "chair"])
    with pytest.raises(ValueError):
        nyuv2.NyuV2(path)


def test_length_and_index_bounds(fake_mat: pathlib.Path) -> None:
    with nyuv2.NyuV2(fake_mat) as data:
        assert len(data) == 2
        with pytest.raises(IndexError):
            data[2]
        with pytest.raises(IndexError):
            data[-1]


# --- Hole 2: the split ------------------------------------------------------


def test_load_split_converts_to_zero_based(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "splits.mat"
    _write_splits(path, train=[3, 1], test=[2])
    split = nyuv2.load_split(path)
    assert split == {"train": [0, 2], "test": [1]}
    assert all(type(i) is int for i in split["train"] + split["test"])


@given(st.lists(st.booleans(), min_size=2, max_size=60))
def test_split_partitions_every_index(in_train: list[bool]) -> None:
    # Hypothesis decides, for each of n images, whether it is train or test.
    n = len(in_train)
    train = [i + 1 for i in range(n) if in_train[i]]
    test = [i + 1 for i in range(n) if not in_train[i]]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "splits.mat"
        _write_splits(path, train, test)
        split = nyuv2.load_split(path)
    assert set(split["train"]).isdisjoint(split["test"])
    assert sorted(split["train"] + split["test"]) == list(range(n))


# --- The real file, when it has been downloaded -----------------------------


@pytest.mark.skipif(
    not (REAL_MAT.exists() and REAL_SPLITS.exists()),
    reason="NYUv2 not downloaded to data/nyuv2/",
)
def test_real_dataset_matches_the_paper() -> None:
    split = nyuv2.load_split(REAL_SPLITS)
    assert (len(split["train"]), len(split["test"])) == (795, 654)
    with nyuv2.NyuV2(REAL_MAT) as data:
        assert len(data) == 1449
        sample = data[split["test"][0]]
    assert sample.rgb.shape == (480, 640, 3)
    assert sample.depth.shape == (480, 640)
    assert 0 < float(sample.depth.max()) < 10.5  # Kinect range, metres
