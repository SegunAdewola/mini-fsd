"""Tests for minifsd.train.yolo: YOLO26n training on Colab (E1a).

ultralytics is not installed locally (ADR 0008), so a fake module stands in
for it. These check our side: the dataset, fresh start against resume, and
the ONNX export. Ultralytics' own training is its business.
"""

import json
import pathlib
import sys
import types
from typing import Any

import pytest

from minifsd.data import boxes
from minifsd.train import yolo


def _split(folder: pathlib.Path, names: list[str]) -> pathlib.Path:
    folder.mkdir(parents=True)
    categories = [
        {"id": i + 1, "name": n} for i, n in enumerate(boxes.CATEGORIES)
    ]
    images = []
    for i, name in enumerate(names, start=1):
        (folder / name).write_bytes(b"x")
        images.append({"id": i, "file_name": name, "width": 320, "height": 240})
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 2,
            "bbox": [10, 20, 30, 40],
            "area": 1200,
            "iscrowd": 0,
        }
    ]
    (folder / "coco.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": categories,
            }
        )
    )
    return folder


def _config(tmp_path: pathlib.Path) -> yolo.Config:
    return yolo.Config(
        train={
            "v10": _split(tmp_path / "v10" / "train", ["a.png", "b.png"]),
            "coco": _split(tmp_path / "coco" / "train", ["1.jpg"]),
        },
        val={"v10": _split(tmp_path / "v10" / "val", ["c.png"])},
        run_dir=tmp_path / "drive" / "runs" / "e1a",
        work_dir=tmp_path / "session" / "yolo",
    )


class FakeYolo:
    """Records what the training code asks of ultralytics.YOLO."""

    calls: list[tuple[str, Any]] = []

    def __init__(self, weights: str) -> None:
        self.weights = str(weights)
        FakeYolo.calls.append(("load", self.weights))

    def train(self, **kwargs: Any) -> None:
        FakeYolo.calls.append(("train", kwargs))
        weights = pathlib.Path(kwargs.get("project", "")) / kwargs.get(
            "name", ""
        )
        if not kwargs.get("resume"):
            (weights / "weights").mkdir(parents=True, exist_ok=True)
            (weights / "weights" / "best.pt").write_bytes(b"pt")
            (weights / "weights" / "last.pt").write_bytes(b"pt")

    def export(self, **kwargs: Any) -> str:
        FakeYolo.calls.append(("export", kwargs))
        onnx = pathlib.Path(self.weights).with_suffix(".onnx")
        onnx.write_bytes(b"onnx")
        return str(onnx)


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYolo]:
    FakeYolo.calls = []
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYolo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return FakeYolo


def test_prepare_builds_both_splits_and_a_data_file(
    tmp_path: pathlib.Path,
) -> None:
    config = _config(tmp_path)
    data = yolo.prepare(config)
    assert data == config.work_dir / "data.yaml"
    root = json.loads(data.read_text())
    assert root["names"] == list(boxes.CATEGORIES)
    train = sorted(
        p.name for p in (config.work_dir / "images" / "train").iterdir()
    )
    assert train == ["coco_1.jpg", "v10_a.png", "v10_b.png"]
    assert (config.work_dir / "labels" / "val" / "v10_c.txt").exists()


def test_prepare_again_reuses_the_built_folder(tmp_path: pathlib.Path) -> None:
    # A rerun after a dropped session must not fail on existing links.
    config = _config(tmp_path)
    yolo.prepare(config)
    assert yolo.prepare(config) == config.work_dir / "data.yaml"


def test_a_fresh_run_trains_from_the_pretrained_model_then_exports(
    tmp_path: pathlib.Path, fake_ultralytics: type[FakeYolo]
) -> None:
    config = _config(tmp_path)
    onnx = yolo.train(config)
    kinds = [kind for kind, _ in fake_ultralytics.calls]
    assert kinds == ["load", "train", "load", "export"]
    assert fake_ultralytics.calls[0][1] == yolo.PRETRAINED
    args = fake_ultralytics.calls[1][1]
    assert args["data"] == str(config.work_dir / "data.yaml")
    assert args["imgsz"] == config.imgsz == 320
    assert args["project"] == str(config.run_dir.parent)
    assert args["name"] == config.run_dir.name
    assert fake_ultralytics.calls[3][1]["format"] == "onnx"
    assert onnx == config.run_dir / "weights" / "best.onnx"


def test_a_run_with_a_checkpoint_resumes_it(
    tmp_path: pathlib.Path, fake_ultralytics: type[FakeYolo]
) -> None:
    config = _config(tmp_path)
    weights = config.run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"pt")
    (weights / "best.pt").write_bytes(b"pt")
    yolo.train(config)
    assert fake_ultralytics.calls[0] == ("load", str(weights / "last.pt"))
    assert fake_ultralytics.calls[1] == ("train", {"resume": True})
    # A new session has an empty session disk: the dataset the checkpoint
    # points at must be rebuilt before resuming.
    assert (config.work_dir / "data.yaml").exists()


def test_a_finished_run_only_exports(
    tmp_path: pathlib.Path,
    fake_ultralytics: type[FakeYolo],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ultralytics refuses to resume a finished run with an AssertionError;
    # the session may have dropped between the last epoch and the export.
    config = _config(tmp_path)
    weights = config.run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"pt")
    (weights / "best.pt").write_bytes(b"pt")

    def finished(self: FakeYolo, **kwargs: Any) -> None:
        raise AssertionError("training to 40 epochs is finished")

    monkeypatch.setattr(FakeYolo, "train", finished)
    assert yolo.train(config) == weights / "best.onnx"
    assert [k for k, _ in fake_ultralytics.calls][-1] == "export"


def test_an_exported_run_is_left_alone(
    tmp_path: pathlib.Path, fake_ultralytics: type[FakeYolo]
) -> None:
    config = _config(tmp_path)
    weights = config.run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.onnx").write_bytes(b"onnx")
    assert yolo.train(config) == weights / "best.onnx"
    assert fake_ultralytics.calls == []


def test_sources_parse_from_the_command_line() -> None:
    assert yolo.parse_sources(["v10=/a/train", "coco=/b/train"]) == {
        "v10": pathlib.Path("/a/train"),
        "coco": pathlib.Path("/b/train"),
    }


def test_the_command_line_passes_workers_through(
    tmp_path: pathlib.Path,
    fake_ultralytics: type[FakeYolo],
) -> None:
    # Explorer jobs get 8 CPUs, Colab 2: the loader count is a flag.
    config = _config(tmp_path)
    yolo.main(
        [
            "--train",
            *(f"{k}={v}" for k, v in config.train.items()),
            "--val",
            *(f"{k}={v}" for k, v in config.val.items()),
            "--run-dir",
            str(config.run_dir),
            "--work-dir",
            str(config.work_dir),
            "--workers",
            "8",
        ]
    )
    args = next(a for kind, a in fake_ultralytics.calls if kind == "train")
    assert args["workers"] == 8
