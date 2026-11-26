"""Trains HydraNet-lite's heads on v10 and COCO people together (H4, E4).

    python -m minifsd.train.hydra --run-dir /scratch/$USER/mini-fsd/runs/e4
        --train $D/v10/train $D/coco-people/train --val $D/v10/val

(one command, wrapped here to fit)

A source with a meta.jsonl is a sim split: every image labels floor, depth
and all four box classes. Any other source is a COCO-style folder: boxes
only, and only obstacle and person, so door and room sign are masked out of
the heatmap loss (minifsd.train.hydra_loss). The backbone (DINOv2-S) stays
frozen; only the heads train. Checkpoints, resume and the log follow
minifsd.train.floor: last.pt every epoch, best/ by validation loss.

Loss: heatmap + 0.1 x size + offset (CenterNet's weights) + floor + depth.
"""

import argparse
import dataclasses
import json
import pathlib
import shutil
import time
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils import data as torch_data
from transformers import Dinov2Model

from minifsd.data import boxes as boxes_lib
from minifsd.perception import centernet, floor_dino, hydra
from minifsd.train import dino, floor, hydra_loss

SIZE_WEIGHT = 0.1  # CenterNet's
LABEL_HEIGHT, LABEL_WIDTH = floor.LABEL_HEIGHT, floor.LABEL_WIDTH
GRID = (
    floor_dino.INPUT_HEIGHT // centernet.STRIDE,
    floor_dino.INPUT_WIDTH // centernet.STRIDE,
)
N_CLASSES = len(boxes_lib.CATEGORIES)
# What a COCO photo labels: its own furniture (as obstacle) and people.
COCO_CLASSES = ("obstacle", "person")

Item = dict[str, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class Config:
    """One E4 run."""

    train: Sequence[pathlib.Path]
    val: Sequence[pathlib.Path]
    run_dir: pathlib.Path
    pretrained: str | None = floor_dino.BACKBONE  # None: tiny (tests)
    epochs: int = 12
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.01
    workers: int = 8
    limit: int | None = None  # per source
    device: str | None = None
    seed: int = 0


def _read(path: pathlib.Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"can't read {path}")
    return image


def flip(item: Item) -> Item:
    """The item mirrored left to right, its box targets exactly so.

    The input is 46 cells wide (322 / 7), so a centre in cell c moves to
    cell 45 - c, and its offset within the cell x to 1 - x.
    """
    out = {k: v.flip(-1) if v.dim() >= 2 else v for k, v in item.items()}
    offset = out["offset"].clone()
    offset[0] = torch.where(out["centres"], 1 - offset[0], offset[0])
    out["offset"] = offset
    return out


class HydraDataset(torch_data.Dataset[Item]):
    """Sim splits and COCO-style folders as HydraNet training items."""

    def __init__(
        self,
        folders: Sequence[pathlib.Path],
        augment: bool,
        limit: int | None = None,
    ) -> None:
        """Lists every image of every folder.

        Args:
            folders: Sim splits (with meta.jsonl) or COCO-style folders.
            augment: Flip half the items.
            limit: Keep only the first N images of each folder.
        """
        self.records: list[tuple[pathlib.Path, dict[str, Any], list[Any], bool]]
        self.records = []
        for folder in folders:
            coco = json.loads((folder / "coco.json").read_text())
            anns: dict[int, list[Any]] = {}
            for a in coco["annotations"]:
                anns.setdefault(a["image_id"], []).append(a)
            is_sim = (folder / "meta.jsonl").exists()
            for image in coco["images"][:limit]:
                self.records.append(
                    (folder, image, anns.get(image["id"], []), is_sim)
                )
        self.augment = augment

    def __len__(self) -> int:
        """The number of images."""
        return len(self.records)

    def __getitem__(self, i: int) -> Item:
        """Image i with its targets and what it labels."""
        folder, image, anns, is_sim = self.records[i]
        bgr = _read(folder / image["file_name"], cv2.IMREAD_COLOR)
        rgb = np.ascontiguousarray(bgr[..., ::-1])
        h, w = rgb.shape[:2]
        sx = floor_dino.INPUT_WIDTH / w
        sy = floor_dino.INPUT_HEIGHT / h
        found = [
            centernet.Box(
                a["category_id"] - 1,
                a["bbox"][0] * sx,
                a["bbox"][1] * sy,
                a["bbox"][2] * sx,
                a["bbox"][3] * sy,
            )
            for a in anns
            if not a["iscrowd"]
        ]
        t = centernet.encode(found, GRID, N_CLASSES)
        floor_label = np.zeros((LABEL_HEIGHT, LABEL_WIDTH), np.int64)
        depth = np.zeros((LABEL_HEIGHT, LABEL_WIDTH), np.float32)
        labelled: tuple[str, ...] = COCO_CLASSES
        if is_sim:
            stem = image["file_name"].replace("_rgb.", "_floor.")
            floor_label = (
                _read(folder / stem, cv2.IMREAD_UNCHANGED) > 0
            ).astype(np.int64)
            mm = _read(
                folder / image["file_name"].replace("_rgb.", "_depth."),
                cv2.IMREAD_UNCHANGED,
            )
            depth = mm.astype(np.float32) / 1000.0
            labelled = boxes_lib.CATEGORIES
        item = {
            "x": torch.from_numpy(floor_dino.preprocess(rgb)[0]),
            "floor": torch.from_numpy(floor_label),
            "depth": torch.from_numpy(depth),
            "has_floor": torch.tensor(is_sim),
            "has_depth": torch.tensor(is_sim),
            "class_mask": torch.tensor(
                [c in labelled for c in boxes_lib.CATEGORIES]
            ),
            "heatmap": torch.from_numpy(t.heatmap),
            "size": torch.from_numpy(t.size),
            "offset": torch.from_numpy(t.offset),
            "centres": torch.from_numpy(t.mask),
        }
        # torch's generator, not numpy's: the loader seeds it per worker.
        if self.augment and torch.rand(()) < 0.5:
            item = flip(item)
        return item


def build_model(pretrained: str | None) -> hydra.HydraNet:
    """HydraNet on DINOv2-S, or on a tiny random backbone (tests)."""
    if pretrained is None:
        backbone = dino.build_model(None).backbone
        return hydra.HydraNet(backbone, neck=16)
    return hydra.HydraNet(Dinov2Model.from_pretrained(pretrained))


def losses(model: hydra.HydraNet, batch: Item) -> dict[str, torch.Tensor]:
    """Each head's loss on one batch, and their weighted total."""
    out = model(batch["x"])
    parts = {
        "heatmap": hydra_loss.focal_loss(
            out.heatmap, batch["heatmap"], batch["class_mask"]
        ),
        "size": hydra_loss.masked_l1(out.size, batch["size"], batch["centres"]),
        "offset": hydra_loss.masked_l1(
            out.offset, batch["offset"], batch["centres"]
        ),
        "floor": hydra_loss.floor_loss(
            out.floor, batch["floor"], batch["has_floor"]
        ),
        "depth": hydra_loss.depth_loss(
            out.log_depth, batch["depth"], batch["has_depth"]
        ),
    }
    parts["total"] = (
        parts["heatmap"]
        + SIZE_WEIGHT * parts["size"]
        + parts["offset"]
        + parts["floor"]
        + parts["depth"]
    )
    return parts


def _save_best(model: hydra.HydraNet, run_dir: pathlib.Path) -> None:
    part = run_dir / f"{floor.BEST}.part"
    if part.exists():
        shutil.rmtree(part)
    model.save(part)
    if (run_dir / floor.BEST).exists():
        shutil.rmtree(run_dir / floor.BEST)
    part.rename(run_dir / floor.BEST)


def _to(batch: Item, device: str) -> Item:
    return {k: v.to(device) for k, v in batch.items()}


def train(config: Config) -> None:
    """Runs (or resumes) E4 training until config.epochs are done."""
    device = floor.device_of(config.device)
    cuda = device == "cuda"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    train_set = HydraDataset(config.train, augment=True, limit=config.limit)
    val_loader = torch_data.DataLoader(
        HydraDataset(config.val, augment=False, limit=config.limit),
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=cuda,
    )
    steps_per_epoch = -(-len(train_set) // config.batch_size)
    total_steps = config.epochs * steps_per_epoch

    model = build_model(config.pretrained).to(device)
    heads = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        heads, lr=config.lr, weight_decay=config.weight_decay
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
        sums: dict[str, float] = {}
        seen = 0
        for step, batch in enumerate(loader):
            batch = _to(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                parts = losses(model, batch)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(parts["total"]).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            n = len(batch["x"])
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item() * n
            seen += n
            if step % floor.PRINT_EVERY == 0:
                print(
                    f"epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {parts['total'].item():.4f}",
                    flush=True,
                )

        model.eval()
        val_sum, val_seen = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                batch = _to(batch, device)
                val_sum += losses(model, batch)["total"].item() * len(
                    batch["x"]
                )
                val_seen += len(batch["x"])
        val_loss = val_sum / val_seen
        if val_loss < best:
            best = val_loss
            _save_best(model, config.run_dir)
        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / seen,
            **{f"train_{k}": v / seen for k, v in sums.items() if k != "total"},
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
    """Trains from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", nargs="+", required=True, type=pathlib.Path)
    parser.add_argument("--val", nargs="+", required=True, type=pathlib.Path)
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
