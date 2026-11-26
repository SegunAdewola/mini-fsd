"""Tests for minifsd.train.da3_predict: DA3 depth on Colab (E3a).

DA3 itself runs on Colab only; a fake model stands in for it here.
"""

import pathlib

import numpy as np

from minifsd.eval import precomputed
from minifsd.train import da3_predict


def _images(folder: pathlib.Path, n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    rgbs = [rng.integers(0, 256, (12, 16, 3), dtype=np.uint8) for _ in range(n)]
    precomputed.export_images(
        [type("S", (), {"rgb": r})() for r in rgbs], folder
    )
    return rgbs


def test_every_image_gets_a_depth_map_under_its_key(
    tmp_path: pathlib.Path,
) -> None:
    rgbs = _images(tmp_path / "in", 3)
    seen: list[tuple[int, ...]] = []

    def fake(rgb: np.ndarray) -> np.ndarray:
        seen.append(rgb.shape)
        return rgb.mean(axis=2) + 1.0  # a "depth" the test can check

    n = da3_predict.predict_folder(tmp_path / "in", tmp_path / "out.npz", fake)
    assert n == 3 and seen == [(12, 16, 3)] * 3
    model = precomputed.PrecomputedDepth(tmp_path / "out.npz")
    for rgb in rgbs:
        expected = 1.0 / (rgb.mean(axis=2) + 1.0)
        np.testing.assert_allclose(model.predict(rgb), expected, rtol=2e-3)


def test_a_finished_file_is_written_whole(tmp_path: pathlib.Path) -> None:
    _images(tmp_path / "in", 2)
    da3_predict.predict_folder(
        tmp_path / "in", tmp_path / "out.npz", lambda rgb: rgb[..., 0] + 1.0
    )
    assert (tmp_path / "out.npz").exists()
    assert not list(tmp_path.glob("*.part*"))
