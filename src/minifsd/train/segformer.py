"""Trains SegFormer-B0 for floor segmentation on v9 (E2a).

    python -m minifsd.train.segformer --train /content/data/v9/train
        --val /content/data/v9/val --run-dir /content/drive/MyDrive/run

(one command, wrapped here to fit)

The loop, checkpoints and resume are minifsd.train.floor's. best/ is a
Hugging Face save_pretrained folder that FloorSegformer loads.

Recipe, from the SegFormer paper's setup scaled to a short fine-tune: AdamW,
learning rate 6e-5 on the ImageNet-pretrained encoder and ten times that on
the new decode head, weight decay 0.01, 20 epochs.
"""

import dataclasses
import pathlib
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from minifsd.perception import floor_segformer
from minifsd.train import floor

PRETRAINED = "nvidia/mit-b0"  # ImageNet-1k encoder, no segmentation head
LABELS = {0: "other", floor_segformer.FLOOR: "floor"}
# The shared loop's names, kept here for callers of this module.
LAST, BEST, LOG = floor.LAST, floor.BEST, floor.LOG
FloorDataset = floor.FloorDataset


@dataclasses.dataclass(frozen=True)
class Config(floor.Config):
    """The E2a run."""

    pretrained: str | None = PRETRAINED
    epochs: int = 20
    batch_size: int = 16
    encoder_lr: float = 6e-5
    head_lr: float = 6e-4


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


def _param_groups(
    model: nn.Module, config: floor.Config
) -> list[dict[str, Any]]:
    return [
        {"params": model.segformer.parameters(), "lr": config.encoder_lr},
        {"params": model.decode_head.parameters(), "lr": config.head_lr},
    ]


def _logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits: torch.Tensor = model(pixel_values=x).logits
    return logits


def _save(model: nn.Module, folder: pathlib.Path) -> None:
    model.save_pretrained(folder)


RECIPE = floor.Recipe(
    build=build_model,
    param_groups=_param_groups,
    logits=_logits,
    preprocess=floor_segformer.preprocess,
    save=_save,
)


def train(config: Config) -> None:
    """Runs (or resumes) the E2a training."""
    floor.train(config, RECIPE)


def main(argv: Sequence[str] | None = None) -> None:
    """Trains from the command line."""
    floor.main(argv, Config, RECIPE, __doc__.split("\n")[0])


if __name__ == "__main__":
    main()
