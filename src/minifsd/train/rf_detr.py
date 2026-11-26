"""Fine-tunes RF-DETR Nano on v10 plus COCO people, on Colab (E1b).

    python -m minifsd.train.rf_detr --run-dir /content/drive/MyDrive/run
        --train v10=/content/data/v10/train coco=/content/data/coco-people/train
        --val v10=/content/data/v10/val coco=/content/data/coco-people/val

(one command, wrapped here to fit)

rfdetr (Apache-2.0, pinned in the notebook) is installed on Colab only and
imported inside train(). Our part, as for YOLO (minifsd.train.yolo):

1. Build Roboflow's COCO layout on the session disk with
   minifsd.train.adapters: train/ and valid/, each with the images and
   _annotations.coco.json. RF-DETR numbers classes by their place in the
   train file's category list, which is minifsd.data.boxes.CATEGORIES.
2. Train RF-DETR Nano into run_dir on Drive. Its trainer rewrites last.ckpt
   every epoch; if one is there, resume from it (ADR 0007).
3. Export the best checkpoint to ONNX beside it, for the harness.

Input is 320 px, as for YOLO26n, so E1 compares the two at the same size
(Nano's own default is 384; any multiple of 32 works).
"""

import argparse
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence

from minifsd.train import adapters, yolo

RESUME = "last.ckpt"
BEST = "checkpoint_best_total.pth"


@dataclasses.dataclass(frozen=True)
class Config:
    """One RF-DETR training run."""

    train: Mapping[str, pathlib.Path]  # prefix -> split folder
    val: Mapping[str, pathlib.Path]
    run_dir: pathlib.Path  # on Drive
    work_dir: pathlib.Path = pathlib.Path("/content/rfdetr")  # session disk
    epochs: int = 20
    resolution: int = 320
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 1e-4  # rfdetr's default for fine-tuning


def prepare(config: Config) -> pathlib.Path:
    """Builds (or reuses) the Roboflow-style dataset; returns its root."""
    valid = config.work_dir / "valid" / adapters.RFDETR_ANNOTATIONS
    if valid.exists():
        return config.work_dir
    adapters.write_coco_folder(config.train, config.work_dir / "train")
    # Written last, so its presence means both splits are complete.
    adapters.write_coco_folder(config.val, config.work_dir / "valid")
    return config.work_dir


def train(config: Config) -> pathlib.Path:
    """Trains, resumes or just exports, as far as the run has got.

    Returns:
        The exported ONNX model in run_dir. rfdetr names it after the model
        (1.12.1: rfdetr-nano.onnx), so any .onnx there means "done".
    """
    exported = sorted(config.run_dir.glob("*.onnx"))
    if exported:
        return exported[0]

    from rfdetr import RFDETRNano  # Colab only

    args: dict[str, object] = {
        "dataset_dir": str(prepare(config)),
        "output_dir": str(config.run_dir),
        "epochs": config.epochs,
        "resolution": config.resolution,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "lr": config.lr,
    }
    last = config.run_dir / RESUME
    if last.exists():
        args["resume"] = str(last)
    RFDETRNano().train(**args)
    best = RFDETRNano.from_checkpoint(str(config.run_dir / BEST))
    onnx = best.export(
        output_dir=str(config.run_dir),
        shape=(config.resolution, config.resolution),
    )
    return pathlib.Path(onnx)


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
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    args = parser.parse_args(argv)
    onnx = train(
        Config(
            train=yolo.parse_sources(args.train),
            val=yolo.parse_sources(args.val),
            run_dir=args.run_dir,
            work_dir=args.work_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    )
    print(f"exported: {onnx}")


if __name__ == "__main__":
    main()
