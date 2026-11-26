"""E3b: a small depth network, taught by sim depth and by Depth Anything V2.

    python -m minifsd.train.depth_student teach --nyu $D/nyuv2 --out $D/teach
    python -m minifsd.train.depth_student train --sim $D/v10/train
        --teacher $D/teach --val $D/v10/val --run-dir $R/e3b

(the second command wrapped here to fit)

The student (minifsd.perception.depth_student) is SegFormer-B0 with one
output channel at 160 x 120. It learns from two kinds of image, with no
human depth label on any real one:

- v10 sim frames, whose exact depth becomes the target 1 / depth;
- NYUv2 train images outside nyuv2-val and nyuv2-fit (harness
  "nyuv2-teach"), labelled by Depth Anything V2-Small (the E3a pick), the
  teacher. `teach` runs it once and saves image and disparity pairs.

Neither target shares a scale with the other or with metres, so the loss is
MiDaS's scale-and-shift invariant one: each prediction is aligned to its
target by least squares, then the mean absolute error is taken. The
teacher's 495 images are repeated so they make half of each epoch.
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional
from torch.utils import data as torch_data
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from minifsd.perception import depth_student
from minifsd.train import floor, segformer

TARGET_HEIGHT, TARGET_WIDTH = 240, 320  # every target is resized to this
TEACHER = "depth-anything/Depth-Anything-V2-Small-hf"  # Hub copy of V5's model
LABELS = {0: "disparity"}

Item = dict[str, torch.Tensor]


def ssi_loss(
    pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """MiDaS's scale-and-shift invariant loss, averaged over images.

    Args:
        pred: (B, 1, h, w) predicted disparity.
        target: (B, H, W) target disparity.
        valid: (B, H, W) bool, pixels with a target.
    """
    if pred.shape[-2:] != target.shape[-2:]:
        pred = functional.interpolate(
            pred, size=target.shape[-2:], mode="bilinear", align_corners=False
        )
    losses = []
    for i in range(len(pred)):
        keep = valid[i]
        if int(keep.sum()) < 2:
            continue
        p, y = pred[i, 0][keep], target[i][keep]
        n = p.numel()
        sp, sy = p.sum(), y.sum()
        det = n * (p * p).sum() - sp * sp
        if float(det.detach().abs()) < 1e-9:  # a flat prediction: no fit
            losses.append((y - y.mean()).abs().mean())
            continue
        s = (n * (p * y).sum() - sp * sy) / det
        t = (sy - s * sp) / n
        losses.append((s * p + t - y).abs().mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()


def _read(path: pathlib.Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"can't read {path}")
    return image


def _to_target(values: np.ndarray, valid: np.ndarray) -> tuple[Any, Any]:
    if values.shape != (TARGET_HEIGHT, TARGET_WIDTH):
        size = (TARGET_WIDTH, TARGET_HEIGHT)
        values = cv2.resize(values, size, interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(
            valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return torch.from_numpy(values.astype(np.float32)), torch.from_numpy(valid)


def export_teacher(
    samples: Iterable[Any],
    predict: Callable[[np.ndarray], np.ndarray],
    out_dir: pathlib.Path,
) -> int:
    """Saves each sample's RGB and the teacher's disparity for it.

    Files are <index>_rgb.png and <index>_disp.npy (float16).

    Returns:
        How many images were written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in samples:
        stem = f"{s.index:05d}"
        cv2.imwrite(str(out_dir / f"{stem}_rgb.png"), s.rgb[..., ::-1])
        disparity = np.asarray(predict(s.rgb), np.float32)
        np.save(out_dir / f"{stem}_disp.npy", disparity.astype(np.float16))
        n += 1
    return n


class TeacherDepth(torch_data.Dataset[Item]):
    """Images with the teacher's disparity, from export_teacher."""

    def __init__(self, folder: pathlib.Path) -> None:
        """Lists the folder's image and disparity pairs."""
        self.rgbs = sorted(folder.glob("*_rgb.png"))

    def __len__(self) -> int:
        """The number of images."""
        return len(self.rgbs)

    def __getitem__(self, i: int) -> Item:
        """Image i as input, its teacher disparity as target."""
        path = self.rgbs[i]
        rgb = np.ascontiguousarray(_read(path, cv2.IMREAD_COLOR)[..., ::-1])
        disp = np.load(
            path.with_name(path.name.replace("_rgb.png", "_disp.npy"))
        )
        disp = disp.astype(np.float32)
        target, valid = _to_target(disp, np.isfinite(disp))
        x = torch.from_numpy(depth_student.preprocess(rgb)[0])
        return {"x": x, "target": target, "valid": valid}


class SimDepth(torch_data.Dataset[Item]):
    """A sim split's frames with 1 / depth as target."""

    def __init__(self, folder: pathlib.Path, limit: int | None = None) -> None:
        """Lists the split's frames (from its coco.json)."""
        coco = json.loads((folder / "coco.json").read_text())
        self.paths = [folder / i["file_name"] for i in coco["images"]][:limit]

    def __len__(self) -> int:
        """The number of frames."""
        return len(self.paths)

    def __getitem__(self, i: int) -> Item:
        """Frame i as input, 1 / depth where there is a reading."""
        path = self.paths[i]
        rgb = np.ascontiguousarray(_read(path, cv2.IMREAD_COLOR)[..., ::-1])
        mm = _read(
            path.with_name(path.name.replace("_rgb.", "_depth.")),
            cv2.IMREAD_UNCHANGED,
        )
        depth = mm.astype(np.float32) / 1000.0
        valid = depth > 0
        disp = np.where(valid, 1.0 / np.maximum(depth, 1e-6), 0.0)
        target, valid_t = _to_target(disp, valid)
        x = torch.from_numpy(depth_student.preprocess(rgb)[0])
        return {"x": x, "target": target, "valid": valid_t}


@dataclasses.dataclass(frozen=True)
class Config:
    """One E3b run."""

    sim: Sequence[pathlib.Path]
    teacher: Sequence[pathlib.Path]
    val: Sequence[pathlib.Path]
    run_dir: pathlib.Path
    pretrained: str | None = segformer.PRETRAINED  # None: tiny (tests)
    epochs: int = 15
    batch_size: int = 32
    encoder_lr: float = 6e-5
    head_lr: float = 6e-4
    weight_decay: float = 0.01
    workers: int = 8
    limit: int | None = None
    device: str | None = None
    seed: int = 0


def build_model(pretrained: str | None) -> SegformerForSemanticSegmentation:
    """SegFormer-B0 with a one-channel head, or a tiny one (tests)."""
    label2id = {v: k for k, v in LABELS.items()}
    if pretrained is None:
        config = SegformerConfig(
            depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64],
            num_attention_heads=[1, 1, 2, 2],
            decoder_hidden_size=32,
            num_labels=1,
            id2label=LABELS,
            label2id=label2id,
        )
        return SegformerForSemanticSegmentation(config)
    model: SegformerForSemanticSegmentation = (
        SegformerForSemanticSegmentation.from_pretrained(
            pretrained, num_labels=1, id2label=LABELS, label2id=label2id
        )
    )
    return model


def _training_set(config: Config) -> torch_data.Dataset[Item]:
    sims = [SimDepth(f, config.limit) for f in config.sim]
    teachers = [TeacherDepth(f) for f in config.teacher]
    n_sim = sum(len(d) for d in sims)
    n_teacher = sum(len(d) for d in teachers)
    repeat = max(1, round(n_sim / n_teacher)) if n_teacher else 0
    parts: list[torch_data.Dataset[Item]] = [*sims, *teachers * repeat]
    return torch_data.ConcatDataset(parts)


def _save_best(
    model: SegformerForSemanticSegmentation, run_dir: pathlib.Path
) -> None:
    part = run_dir / f"{floor.BEST}.part"
    if part.exists():
        shutil.rmtree(part)
    model.save_pretrained(part)
    if (run_dir / floor.BEST).exists():
        shutil.rmtree(run_dir / floor.BEST)
    part.rename(run_dir / floor.BEST)


def _loss(model: SegformerForSemanticSegmentation, batch: Item) -> torch.Tensor:
    pred = model(pixel_values=batch["x"]).logits
    return ssi_loss(pred, batch["target"], batch["valid"])


def train(config: Config) -> None:
    """Runs (or resumes) E3b training until config.epochs are done."""
    device = floor.device_of(config.device)
    cuda = device == "cuda"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    train_set = _training_set(config)
    val_loader = torch_data.DataLoader(
        torch_data.ConcatDataset(
            [SimDepth(f, config.limit) for f in config.val]
        ),
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=cuda,
    )
    steps_per_epoch = -(-len(train_set) // config.batch_size)
    total_steps = config.epochs * steps_per_epoch

    model = build_model(config.pretrained).to(device)
    optimiser = torch.optim.AdamW(
        [
            {"params": model.segformer.parameters(), "lr": config.encoder_lr},
            {"params": model.decode_head.parameters(), "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: max(0.0, 1.0 - step / total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    start, best = 0, float("inf")
    last = config.run_dir / floor.LAST
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["epoch"] + 1, state["best_val_loss"]
        print(f"resuming after epoch {state['epoch']}")

    for epoch in range(start, config.epochs):
        began = time.monotonic()
        model.train()
        order = torch.Generator().manual_seed(config.seed * 1000 + epoch)
        loader = torch_data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            generator=order,
            num_workers=config.workers,
            pin_memory=cuda,
        )
        total, seen = 0.0, 0
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = _loss(model, batch)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item() * len(batch["x"])
            seen += len(batch["x"])
            if step % floor.PRINT_EVERY == 0:
                print(
                    f"epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )
        model.eval()
        val_total, val_seen = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_total += _loss(model, batch).item() * len(batch["x"])
                val_seen += len(batch["x"])
        val_loss = val_total / val_seen
        if val_loss < best:
            best = val_loss
            _save_best(model, config.run_dir)
        row = {
            "epoch": epoch,
            "train_loss": total / seen,
            "val_loss": val_loss,
            "best_val_loss": best,
            "seconds": round(time.monotonic() - began, 1),
            "device": floor.device_name(device),
        }
        with (config.run_dir / floor.LOG).open("a") as f:
            f.write(json.dumps(row) + "\n")
        floor.save_atomic(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "schedule": schedule.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best,
            },
            last,
        )
        print(json.dumps(row), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """`teach` labels nyuv2-teach with the teacher; `train` trains."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    teach = commands.add_parser("teach")
    teach.add_argument("--nyu", required=True, type=pathlib.Path)
    teach.add_argument("--out", required=True, type=pathlib.Path)
    teach.add_argument("--teacher", default=TEACHER)
    teach.add_argument("--device", default="cuda")
    fit = commands.add_parser("train")
    fit.add_argument("--sim", nargs="+", required=True, type=pathlib.Path)
    fit.add_argument("--teacher", nargs="+", required=True, type=pathlib.Path)
    fit.add_argument("--val", nargs="+", required=True, type=pathlib.Path)
    fit.add_argument("--run-dir", required=True, type=pathlib.Path)
    fit.add_argument("--epochs", type=int, default=Config.epochs)
    fit.add_argument("--batch-size", type=int, default=Config.batch_size)
    fit.add_argument("--workers", type=int, default=Config.workers)
    args = parser.parse_args(argv)

    if args.command == "teach":
        from minifsd.eval import harness
        from minifsd.perception import depth_anything

        harness.NYU = args.nyu
        model = depth_anything.DepthAnything(
            pathlib.Path(args.teacher), device=args.device
        )
        n = export_teacher(
            harness.load_dataset("nyuv2-teach"), model.predict, args.out
        )
        print(f"{n} teacher images in {args.out}")
        return
    train(
        Config(
            sim=args.sim,
            teacher=args.teacher,
            val=args.val,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
