"""Trains a floor head on frozen DINOv2 ViT-S/14 features (E2b).

    python -m minifsd.train.dino --train /content/data/v9/train
        --val /content/data/v9/val --run-dir /content/drive/MyDrive/run

(one command, wrapped here to fit)

The loop, checkpoints and resume are minifsd.train.floor's; the model is
minifsd.perception.floor_dino. Only the head trains (6,146 weights), so
it learns fast: AdamW at 1e-3 for 10 epochs, batch 32. best/ holds the
backbone too, so the harness loads it offline.
"""

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from transformers import Dinov2Config, Dinov2Model

from minifsd.perception import floor_dino
from minifsd.train import floor


@dataclasses.dataclass(frozen=True)
class Config(floor.Config):
    """The E2b run."""

    pretrained: str | None = floor_dino.BACKBONE
    epochs: int = 10
    batch_size: int = 32
    head_lr: float = 1e-3


def build_model(pretrained: str | None) -> floor_dino.DinoFloor:
    """A frozen DINOv2 with a fresh head.

    Args:
        pretrained: A Hugging Face name or folder for the backbone, or None
            for a tiny random one that needs no download (tests).
    """
    if pretrained is None:
        backbone = Dinov2Model(
            Dinov2Config(
                hidden_size=32,
                num_hidden_layers=floor_dino.LAYERS,
                num_attention_heads=2,
                intermediate_size=64,
                patch_size=floor_dino.PATCH,
                image_size=floor_dino.PATCH * 4,
            )
        )
    else:
        backbone = Dinov2Model.from_pretrained(pretrained)
    return floor_dino.DinoFloor(backbone)


def _param_groups(
    model: nn.Module, config: floor.Config
) -> list[dict[str, Any]]:
    head = [
        p for n, p in model.named_parameters() if not n.startswith("backbone.")
    ]
    return [{"params": head, "lr": config.head_lr}]


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits: torch.Tensor = model(x)
    return logits


def _save(model: nn.Module, folder: pathlib.Path) -> None:
    assert isinstance(model, floor_dino.DinoFloor)
    model.save(folder)


RECIPE = floor.Recipe(
    build=build_model,
    param_groups=_param_groups,
    logits=_logits,
    preprocess=floor_dino.preprocess,
    save=_save,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Trains from the command line."""
    floor.main(argv, Config, RECIPE, __doc__.split("\n")[0])


if __name__ == "__main__":
    main()
