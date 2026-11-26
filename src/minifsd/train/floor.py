"""The floor segmentation training loop, shared by E2a and E2b.

Each model supplies a Recipe: how to build it, which weights train, how to
turn an image into its input, and how to save it. The loop is the same for
all, so the models differ only in what E2 compares.

Made to survive Colab dropping the session (ADR 0007). After every epoch it
writes, inside run_dir:

- last.pt: model, optimiser, schedule and scaler state. A new call with the
  same run_dir resumes from it, so a lost session costs one epoch at most.
- best/: the model with the highest floor IoU on val so far, saved by the
  recipe; the harness loads it.
- log.jsonl: one line per epoch: loss, val IoU, seconds, device.

The loss is cross-entropy on the logits scaled up to the label's size
(bilinear), the same as SegFormer's own loss in transformers. The learning
rate falls linearly to 0 (poly, power 1). Mixed precision on a GPU. The only
augmentation is a horizontal flip: v9 already randomises looks, lights and
the camera.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import time
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from torch.utils import data as torch_data

from minifsd.eval import metrics
from minifsd.perception import floor_segformer
from minifsd.train import adapters

LAST = "last.pt"
BEST = "best"
LOG = "log.jsonl"
PRINT_EVERY = 100  # steps
LABEL_HEIGHT = 240  # v9's camera; labels are scored at this size
LABEL_WIDTH = 320


@dataclasses.dataclass(frozen=True)
class Config:
    """One training run; each model's module sets its own defaults."""

    train: pathlib.Path
    val: pathlib.Path
    run_dir: pathlib.Path
    pretrained: str | None = None  # None: a tiny random model (tests)
    epochs: int = 20
    batch_size: int = 16
    encoder_lr: float = 6e-5
    head_lr: float = 6e-4
    weight_decay: float = 0.01
    workers: int = 2  # Colab's free tier has 2 CPU cores
    limit: int | None = None  # train on the first N frames only
    device: str | None = None  # None: cuda, else mps, else cpu
    seed: int = 0


Preprocess = Callable[[np.ndarray], np.ndarray]


@dataclasses.dataclass(frozen=True)
class Recipe:
    """What the loop needs to know about one model."""

    build: Callable[[str | None], nn.Module]  # from Config.pretrained
    param_groups: Callable[[nn.Module, Config], list[dict[str, Any]]]
    logits: Callable[[nn.Module, torch.Tensor], torch.Tensor]  # (B, 2, h, w)
    preprocess: Preprocess  # (H, W, 3) uint8 RGB to (1, 3, h, w) float32
    save: Callable[[nn.Module, pathlib.Path], None]  # writes a folder


def _read(path: pathlib.Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"can't read {path}")
    return image


class FloorDataset(torch_data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A v9 split as (image, label) tensors for the model."""

    def __init__(
        self,
        folder: pathlib.Path,
        augment: bool,
        limit: int | None = None,
        preprocess: Preprocess = floor_segformer.preprocess,
    ) -> None:
        """Lists the split's frames.

        Args:
            folder: A v9 split folder.
            augment: Flip half the frames left to right.
            limit: Keep only the first N frames.
            preprocess: The model's image to input function.
        """
        self.pairs = adapters.floor_pairs(folder)[:limit]
        self.augment = augment
        self.preprocess = preprocess

    def __len__(self) -> int:
        """The number of frames."""
        return len(self.pairs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Frame i as a (3, h, w) model input and a (240, 320) 0/1 label."""
        rgb_path, mask_path = self.pairs[i]
        rgb = _read(rgb_path, cv2.IMREAD_COLOR)[..., ::-1]
        mask: np.ndarray = _read(mask_path, cv2.IMREAD_UNCHANGED) > 0
        expected = (LABEL_HEIGHT, LABEL_WIDTH)
        if mask.shape != expected:
            raise ValueError(f"{mask_path}: {mask.shape}, expected {expected}")
        # torch's generator, not numpy's: the loader seeds it per worker.
        if self.augment and torch.rand(()) < 0.5:
            rgb, mask = rgb[:, ::-1], mask[:, ::-1]
        x = self.preprocess(np.ascontiguousarray(rgb))[0]
        y = np.ascontiguousarray(mask).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


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


def _save_best(model: nn.Module, recipe: Recipe, run_dir: pathlib.Path) -> None:
    part = run_dir / f"{BEST}.part"
    if part.exists():
        shutil.rmtree(part)
    recipe.save(model, part)
    if (run_dir / BEST).exists():
        shutil.rmtree(run_dir / BEST)
    part.rename(run_dir / BEST)


def loss_of(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with the logits scaled up to the labels' size."""
    big = functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )
    return functional.cross_entropy(big, labels)


def evaluate_iou(
    model: nn.Module,
    recipe: Recipe,
    loader: torch_data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> float:
    """Floor IoU over a whole split: total intersection over total union."""
    model.eval()
    inter = union = 0
    with torch.inference_mode():
        for x, y in loader:
            logits = recipe.logits(model, x.to(device)).float().cpu()
            for one, truth in zip(logits.numpy(), y.numpy(), strict=True):
                h, w = truth.shape
                pred = floor_segformer.mask_from_logits(one, h, w)
                i, u = metrics.intersection_union(pred, truth.astype(bool))
                inter, union = inter + i, union + u
    model.train()
    return inter / union if union else 1.0


def train(config: Config, recipe: Recipe) -> None:
    """Runs (or resumes) training until config.epochs are done."""
    device = _device(config.device)
    cuda = device == "cuda"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)  # the head's starting weights

    train_set = FloorDataset(
        config.train, True, config.limit, recipe.preprocess
    )
    val_set = FloorDataset(config.val, False, None, recipe.preprocess)
    val_loader = torch_data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=cuda,
    )
    steps_per_epoch = -(-len(train_set) // config.batch_size)
    total_steps = config.epochs * steps_per_epoch

    model = recipe.build(config.pretrained).to(device)
    optimiser = torch.optim.AdamW(
        recipe.param_groups(model, config), weight_decay=config.weight_decay
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
        )
        loss_sum, seen = 0.0, 0
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = loss_of(recipe.logits(model, x), y)
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

        val_iou = evaluate_iou(model, recipe, val_loader, device)
        if val_iou > best_iou:
            best_iou = val_iou
            _save_best(model, recipe, config.run_dir)
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


def main(
    argv: Sequence[str] | None,
    config_type: type[Config],
    recipe: Recipe,
    description: str,
) -> None:
    """Trains from the command line; the flags mirror Config.

    Args:
        argv: Arguments, or None for sys.argv.
        config_type: The model's Config subclass, which holds its defaults.
        recipe: The model's recipe.
        description: For --help.
    """
    defaults = config_type(
        train=pathlib.Path(), val=pathlib.Path(), run_dir=pathlib.Path()
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", required=True, type=pathlib.Path)
    parser.add_argument("--val", required=True, type=pathlib.Path)
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    train(
        config_type(
            train=args.train,
            val=args.val,
            run_dir=args.run_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            workers=args.workers,
            limit=args.limit,
        ),
        recipe,
    )
