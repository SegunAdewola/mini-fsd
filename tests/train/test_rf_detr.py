"""Tests for minifsd.train.rf_detr: RF-DETR Nano training on Colab (E1b).

rfdetr is installed on Colab only, so a fake module stands in for it. These
check our side: the COCO folders, fresh start against resume, and the ONNX
export from the best checkpoint.
"""

import json
import pathlib
import sys
import types
from typing import Any

import pytest

from minifsd.data import boxes
from minifsd.train import rf_detr


def _split(folder: pathlib.Path, names: list[str]) -> pathlib.Path:
    folder.mkdir(parents=True)
    categories = [
        {"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)
    ]
    images = [
        {"id": i, "file_name": n, "width": 320, "height": 240}
        for i, n in enumerate(names, start=1)
    ]
    for n in names:
        (folder / n).write_bytes(b"x")
    (folder / "coco.json").write_text(
        json.dumps(
            {"images": images, "annotations": [], "categories": categories}
        )
    )
    return folder


def _config(tmp_path: pathlib.Path) -> rf_detr.Config:
    return rf_detr.Config(
        train={"v10": _split(tmp_path / "v10" / "train", ["a.png"])},
        val={"v10": _split(tmp_path / "v10" / "val", ["b.png"])},
        run_dir=tmp_path / "drive" / "runs" / "e1b",
        work_dir=tmp_path / "session" / "rfdetr",
    )


class FakeNano:
    """Records what the training code asks of rfdetr.RFDETRNano."""

    calls: list[tuple[str, Any]] = []

    def train(self, **kwargs: Any) -> None:
        FakeNano.calls.append(("train", kwargs))
        out = pathlib.Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "checkpoint_best_total.pth").write_bytes(b"pth")

    @classmethod
    def from_checkpoint(cls, path: str) -> "FakeNano":
        FakeNano.calls.append(("load", str(path)))
        return cls()

    def export(self, **kwargs: Any) -> pathlib.Path:
        FakeNano.calls.append(("export", kwargs))
        # rfdetr 1.12.1 names the file after the model (seen on Explorer).
        onnx = pathlib.Path(kwargs["output_dir"]) / "rfdetr-nano.onnx"
        onnx.write_bytes(b"onnx")
        return onnx


@pytest.fixture
def fake_rfdetr(monkeypatch: pytest.MonkeyPatch) -> type[FakeNano]:
    FakeNano.calls = []
    module = types.ModuleType("rfdetr")
    module.RFDETRNano = FakeNano  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rfdetr", module)
    return FakeNano


def test_prepare_writes_roboflow_train_and_valid_folders(
    tmp_path: pathlib.Path,
) -> None:
    config = _config(tmp_path)
    assert rf_detr.prepare(config) == config.work_dir
    for split, image in (("train", "v10_a.png"), ("valid", "v10_b.png")):
        folder = config.work_dir / split
        coco = json.loads((folder / "_annotations.coco.json").read_text())
        assert [i["file_name"] for i in coco["images"]] == [image]
        assert (folder / image).exists()
    assert rf_detr.prepare(config) == config.work_dir  # a rerun reuses it


def test_a_fresh_run_trains_at_320_then_exports_the_best_checkpoint(
    tmp_path: pathlib.Path, fake_rfdetr: type[FakeNano]
) -> None:
    config = _config(tmp_path)
    onnx = rf_detr.train(config)
    kinds = [kind for kind, _ in fake_rfdetr.calls]
    assert kinds == ["train", "load", "export"]
    args = fake_rfdetr.calls[0][1]
    assert args["dataset_dir"] == str(config.work_dir)
    assert args["output_dir"] == str(config.run_dir)
    assert args["resolution"] == 320
    assert "resume" not in args
    assert fake_rfdetr.calls[1][1] == str(
        config.run_dir / "checkpoint_best_total.pth"
    )
    assert fake_rfdetr.calls[2][1]["shape"] == (320, 320)
    assert onnx == config.run_dir / "rfdetr-nano.onnx"


def test_a_run_with_a_checkpoint_resumes_it(
    tmp_path: pathlib.Path, fake_rfdetr: type[FakeNano]
) -> None:
    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "last.ckpt").write_bytes(b"ckpt")
    rf_detr.train(config)
    assert fake_rfdetr.calls[0][1]["resume"] == str(
        config.run_dir / "last.ckpt"
    )


def test_an_exported_run_is_left_alone(
    tmp_path: pathlib.Path, fake_rfdetr: type[FakeNano]
) -> None:
    config = _config(tmp_path)
    config.run_dir.mkdir(parents=True)
    (config.run_dir / "rfdetr-nano.onnx").write_bytes(b"onnx")
    assert rf_detr.train(config) == config.run_dir / "rfdetr-nano.onnx"
    assert fake_rfdetr.calls == []
