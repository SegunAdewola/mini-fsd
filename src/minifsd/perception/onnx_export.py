"""Exports the picked floor and depth models to ONNX (M4.4, cv env only).

    python -m minifsd.perception.onnx_export KIND FOLDER OUT

KIND is floor or depth; FOLDER is the trained model; OUT the .onnx file,
e.g. floor data/models/floor-dinov2-s-v10 data/models/floor-dinov2-s-v10.onnx.

Each graph is fixed-size pixels in, logits out: the input is exactly what
the torch wrapper's preprocess makes, and everything after the logits
(upsampling, thresholds) stays in numpy, in minifsd.perception.onnx_models.
A fixed size keeps the graph simple and is all the camera needs.
"""

import argparse
import pathlib
from collections.abc import Sequence

import torch
from torch import nn

from minifsd.perception import depth_student, floor_dino


class _Logits(nn.Module):
    """A Hugging Face model's logits as a plain tensor-out module."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.model(pixel_values=pixel_values).logits
        return logits


def _export(
    model: nn.Module, height: int, width: int, out: pathlib.Path
) -> pathlib.Path:
    model.eval()
    example = torch.zeros(1, 3, height, width)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (example,),
        str(out),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamo=True,
        external_data=False,  # one self-contained file
    )
    return out


def export_floor(folder: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Writes a DinoFloor folder (floor_dino.DinoFloor.save) as ONNX."""
    model = floor_dino.DinoFloor.load(folder)
    return _export(model, floor_dino.INPUT_HEIGHT, floor_dino.INPUT_WIDTH, out)


def export_depth(folder: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Writes a depth student folder (save_pretrained) as ONNX."""
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(folder)
    return _export(
        _Logits(model),
        depth_student.INPUT_HEIGHT,
        depth_student.INPUT_WIDTH,
        out,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Exports one model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("kind", choices=["floor", "depth"])
    parser.add_argument("folder", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    args = parser.parse_args(argv)
    export = export_floor if args.kind == "floor" else export_depth
    print(f"wrote {export(args.folder, args.out)}")


if __name__ == "__main__":
    main()
