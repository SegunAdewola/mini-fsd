"""Fine-tunes YOLO26n on v10 plus COCO people, on Colab (E1a).

    python -m minifsd.train.yolo --run-dir /content/drive/MyDrive/run
        --train v10=/content/data/v10/train coco=/content/data/coco-people/train
        --val v10=/content/data/v10/val coco=/content/data/coco-people/val

(one command, wrapped here to fit)

ultralytics is AGPL-3.0 and is installed on Colab only (ADR 0008); it is
imported inside train(), so the rest of minifsd never loads it. Our part:

1. Build the YOLO dataset on the session disk with minifsd.train.adapters
   (symlinks, seconds), or reuse it if this session built it already.
2. Train from the COCO-pretrained yolo26n.pt into run_dir on Drive. If
   run_dir holds a last.pt, resume it instead: Ultralytics saves one every
   epoch, so a dropped session costs at most an epoch (ADR 0007).
3. Export best.pt to ONNX beside it. The harness and the demo run that file
   with ONNX Runtime; no Ultralytics code runs on the M1.

Images go in at 320 px, the sim camera's width: the latency budget is tight,
and real photos are shrunk to the camera's size anyway. COCO images have
doors and room signs nobody labeled; the first run accepts that (E1a).
"""

import argparse
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence

from minifsd.train import adapters

PRETRAINED = "yolo26n.pt"  # COCO-pretrained; ultralytics downloads it
DATA_FILE = "data.yaml"


@dataclasses.dataclass(frozen=True)
class Config:
    """One YOLO training run."""

    train: Mapping[str, pathlib.Path]  # prefix -> split folder
    val: Mapping[str, pathlib.Path]
    run_dir: pathlib.Path  # on Drive: weights/, results.csv
    work_dir: pathlib.Path = pathlib.Path("/content/yolo")  # session disk
    epochs: int = 40
    imgsz: int = 320
    batch: int = 64
    workers: int = 2  # Colab's free tier has 2 CPU cores
    seed: int = 0


def prepare(config: Config) -> pathlib.Path:
    """Builds (or reuses) the YOLO dataset; returns its data file."""
    data = config.work_dir / DATA_FILE
    if data.exists():
        return data
    adapters.write_yolo(config.train, config.work_dir / "images" / "train")
    adapters.write_yolo(config.val, config.work_dir / "images" / "val")
    # Written last, so its presence means both splits are complete.
    return adapters.write_yolo_yaml(config.work_dir, ["train"], ["val"])


def _weights(config: Config) -> pathlib.Path:
    return config.run_dir / "weights"


def train(config: Config) -> pathlib.Path:
    """Trains, resumes or just exports, as far as the run has got.

    Returns:
        The exported ONNX model, run_dir/weights/best.onnx.
    """
    weights = _weights(config)
    onnx = weights / "best.onnx"
    if onnx.exists():
        return onnx  # done, exported

    from ultralytics import YOLO  # AGPL-3.0, Colab only (ADR 0008)

    last = weights / "last.pt"
    if last.exists():
        try:
            YOLO(str(last)).train(resume=True)
        except AssertionError as finished:
            # Ultralytics' way of saying all epochs ran; the session
            # dropped before the export.
            print(f"not resuming: {finished}")
    else:
        data = prepare(config)
        YOLO(PRETRAINED).train(
            data=str(data),
            epochs=config.epochs,
            imgsz=config.imgsz,
            batch=config.batch,
            workers=config.workers,
            seed=config.seed,
            project=str(config.run_dir.parent),
            name=config.run_dir.name,
            exist_ok=True,  # the run folder is ours; reuse it
            cache="ram",  # decode each image once: 2 CPUs are the bottleneck
            plots=False,
        )
    YOLO(str(weights / "best.pt")).export(format="onnx", imgsz=config.imgsz)
    return onnx


def parse_sources(specs: Sequence[str]) -> dict[str, pathlib.Path]:
    """Arguments of the form name=folder, as a prefix to folder mapping."""
    pairs = (spec.split("=", 1) for spec in specs)
    return {name: pathlib.Path(folder) for name, folder in pairs}


def main(argv: Sequence[str] | None = None) -> None:
    """Trains from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+", required=True)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--work-dir", type=pathlib.Path, default=Config.work_dir
    )
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch", type=int, default=Config.batch)
    args = parser.parse_args(argv)
    onnx = train(
        Config(
            train=parse_sources(args.train),
            val=parse_sources(args.val),
            run_dir=args.run_dir,
            work_dir=args.work_dir,
            epochs=args.epochs,
            batch=args.batch,
        )
    )
    print(f"exported: {onnx}")


if __name__ == "__main__":
    main()
