"""Trains SegFormer-B0 for floor segmentation on v9 (E2a).

    python -m minifsd.train.segformer --train /content/data/v9/train
        --val /content/data/v9/val --run-dir /content/drive/MyDrive/run

(one command, wrapped here to fit)

Made to survive Colab dropping the session (ADR 0007). After every epoch it
writes, inside run_dir:

- last.pt: model, optimiser, schedule and scaler state. A new call with the
  same run_dir resumes from it, so a lost session costs one epoch at most.
- best/: the model with the highest floor IoU on val so far, as a Hugging
  Face save_pretrained folder; FloorSegformer and the harness load it.
- log.jsonl: one line per epoch: loss, val IoU, seconds, device.

Recipe, from the SegFormer paper's setup scaled to a short fine-tune: AdamW,
learning rate 6e-5 on the ImageNet-pretrained encoder and ten times that on
the new decode head, falling linearly to 0 (poly, power 1), weight decay
0.01. Mixed precision on a GPU. The only augmentation is a horizontal flip:
v9 already randomises looks, lights and the camera.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils import data as torch_data
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from minifsd.eval import metrics
from minifsd.perception import floor_segformer
from minifsd.train import adapters

PRETRAINED = "nvidia/mit-b0"  # ImageNet-1k encoder, no segmentation head
LAST = "last.pt"
BEST = "best"
LOG = "log.jsonl"
LABELS = {0: "other", floor_segformer.FLOOR: "floor"}
PRINT_EVERY = 100  # steps


@dataclasses.dataclass(frozen=True)
class Config:
    """One training run. The defaults are the E2a run."""

    train: pathlib.Path
    val: pathlib.Path
    run_dir: pathlib.Path
    pretrained: str | None = PRETRAINED  # None: a tiny random model (tests)
    epochs: int = 20
    batch_size: int = 16
    encoder_lr: float = 6e-5
    head_lr_scale: float = 10.0
    weight_decay: float = 0.01
    workers: int = 2  # Colab's free tier has 2 CPU cores
    limit: int | None = None  # train on the first N frames only
    device: str | None = None  # None: cuda, else mps, else cpu
    seed: int = 0


def _read(path: pathlib.Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"can't read {path}")
    return image


class FloorDataset(torch_data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A v9 split as (image, label) tensors for the model."""

    def __init__(
        self, folder: pathlib.Path, augment: bool, limit: int | None = None
    ) -> None:
        """Lists the split's frames.

        Args:
            folder: A v9 split folder.
            augment: Flip half the frames left to right.
            limit: Keep only the first N frames.
        """
        self.pairs = adapters.floor_pairs(folder)[:limit]
        self.augment = augment

    def __len__(self) -> int:
        """The number of frames."""
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame i as a (3, H, W) float image and an (H, W) 0/1 label."""
        rgb_path, mask_path = self.pairs[i]
        rgb = _read(rgb_path, cv2.IMREAD_COLOR)[..., ::-1]
        mask: np.ndarray = _read(mask_path, cv2.IMREAD_UNCHANGED) > 0
        expected = (floor_segformer.INPUT_HEIGHT, floor_segformer.INPUT_WIDTH)
        if mask.shape != expected:
            raise ValueError(f"{mask_path}: {mask.shape}, expected {expected}")
        # torch's generator, not numpy's: the loader seeds it per worker.
        if self.augment and torch.rand(()) < 0.5:
            rgb, mask = rgb[:, ::-1], mask[:, ::-1]
        x = floor_segformer.preprocess(np.ascontiguousarray(rgb))[0]
        y = np.ascontiguousarray(mask).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def build_model(pretrained: str | None) -> SegformerForSemanticSegmentation:
    """SegFormer with a fresh two-class head.

    Args:
        pretrained: A Hugging Face name or folder for the encoder, or None
            for a tiny random model that needs no download (tests).
    """
    label2id = {v: k for k, v in LABELS.items()}
    if pretrained is None:
        config = SegformerConfig(
            depths=[1, 1, 1, 1],
            hidden_sizes=[8, 16, 32, 64],
            num_attention_heads=[1, 1, 2, 2],
            decoder_hidden_size=32,
            num_labels=len(LABELS),
            id2label=LABELS,
            label2id=label2id,
        )
        return SegformerForSemanticSegmentation(config)
    model: SegformerForSemanticSegmentation = (
        SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=len(LABELS),
            id2label=LABELS,
            label2id=label2id,
        )
    )
    return model


def _device(choice: str | None) -> str:
    if choice is not None:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _device_name(device: str) -> str:
    if device == "cuda":
        return str(torch.cuda.get_device_name())
    return device


def _save_atomic(state: dict[str, Any], path: pathlib.Path) -> None:
    part = path.with_name(path.name + ".part")
    torch.save(state, part)
    os.replace(part, path)  # a dropped session never leaves half a file


def _save_best(
    model: SegformerForSemanticSegmentation, run_dir: pathlib.Path
) -> None:
    part = run_dir / f"{BEST}.part"
    if part.exists():
        shutil.rmtree(part)
    model.save_pretrained(part)
    if (run_dir / BEST).exists():
        shutil.rmtree(run_dir / BEST)
    part.rename(run_dir / BEST)


def evaluate_iou(
    model: SegformerForSemanticSegmentation,
    loader: torch_data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> float:
    """Floor IoU over a whole split: total intersection over total union."""
    model.eval()
    inter = union = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = model(pixel_values=x.to(device)).logits.float().cpu()
            for one, truth in zip(logits.numpy(), y.numpy(), strict=True):
                h, w = truth.shape
                pred = floor_segformer.mask_from_logits(one, h, w)
                i, u = metrics.intersection_union(pred, truth.astype(bool))
                inter, union = inter + i, union + u
    model.train()
    return inter / union if union else 1.0


def train(config: Config) -> None:
    """Runs (or resumes) training until config.epochs are done."""
    device = _device(config.device)
    cuda = device == "cuda"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)  # the head's starting weights

    train_set = FloorDataset(config.train, augment=True, limit=config.limit)
    val_set = FloorDataset(config.val, augment=False)
    val_loader = torch_data.DataLoader(
        val_set,
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
            {
                "params": model.decode_head.parameters(),
                "lr": config.encoder_lr * config.head_lr_scale,
            },
        ],
        weight_decay=config.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: max(0.0, 1.0 - step / total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    start, best_iou = 0, -1.0
    last = config.run_dir / LAST
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        start, best_iou = state["epoch"] + 1, state["best_iou"]
        print(
            f"resuming after epoch {state['epoch']} (best IoU {best_iou:.4f})"
        )

    model.train()
    for epoch in range(start, config.epochs):
        began = time.monotonic()
        # Seeded per epoch, so a resumed run shuffles as an unbroken one.
        order = torch.Generator().manual_seed(config.seed * 1000 + epoch)
        loader = torch_data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            generator=order,
            num_workers=config.workers,
            pin_memory=cuda,
            drop_last=False,
        )
        loss_sum, seen = 0.0, 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = model(pixel_values=x, labels=y).loss
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            value = loss.item()  # detached; syncs the GPU once a step
            loss_sum += value * len(x)
            seen += len(x)
            if step % PRINT_EVERY == 0:
                print(
                    f"epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {value:.4f}",
                    flush=True,
                )

        val_iou = evaluate_iou(model, val_loader, device)
        if val_iou > best_iou:
            best_iou = val_iou
            _save_best(model, config.run_dir)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "val_iou": val_iou,
            "best_iou": best_iou,
            "seconds": round(time.monotonic() - began, 1),
            "device": _device_name(device),
        }
        with (config.run_dir / LOG).open("a") as f:
            f.write(json.dumps(row) + "\n")
        _save_atomic(
            {
                "model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "schedule": schedule.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou,
                "config": {
                    k: str(v) for k, v in dataclasses.asdict(config).items()
                },
            },
            last,
        )
        print(json.dumps(row), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    """Trains from the command line; the flags mirror Config."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", required=True, type=pathlib.Path)
    parser.add_argument("--val", required=True, type=pathlib.Path)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--workers", type=int, default=Config.workers)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    train(
        Config(
            train=args.train,
            val=args.val,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            limit=args.limit,
        )
    )


if __name__ == "__main__":
    main()
