"""Exports the picked floor and depth models to ONNX (M4.4, cv env only).

    python -m minifsd.perception.onnx_export KIND SOURCE OUT

KIND is floor or depth (SOURCE is the trained model's folder) or int8
(SOURCE is an FP32 .onnx file, E6.1); OUT is the .onnx file to write, e.g.
floor data/models/floor-dinov2-s-v10 data/models/floor-dinov2-s-v10.onnx.

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


def quantize(fp32: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """Writes an INT8 copy of an ONNX model (E6.1).

    Dynamic quantization: weights stored as 8-bit integers, activations
    scaled on the fly at run time, so it needs no calibration images. It
    suits the ViT floor model best, which is mostly matrix products, the
    operations dynamic INT8 speeds up. ONNX Runtime's pre-processing step
    (shape inference and graph clean-up) runs first, as its docs advise.

    Args:
        fp32: The FP32 model, from export_floor, export_depth or YOLO's own
            export.
        out: Where to write the INT8 model.

    Returns:
        out.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out.parent.mkdir(parents=True, exist_ok=True)
    prepared = out.with_name(out.stem + ".prep.onnx")
    try:
        quant_pre_process(str(fp32), str(prepared))
        quantize_dynamic(str(prepared), str(out), weight_type=QuantType.QInt8)
    finally:
        prepared.unlink(missing_ok=True)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Exports one model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("kind", choices=["floor", "depth", "int8"])
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    args = parser.parse_args(argv)
    export = {"floor": export_floor, "depth": export_depth, "int8": quantize}
    print(f"wrote {export[args.kind](args.source, args.out)}")


if __name__ == "__main__":
    main()
