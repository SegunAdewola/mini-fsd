"""Tests for ONNX export and the ONNX Runtime floor and depth models (M4.4).

Tiny random models, exported and run on CPU. Needs torch, onnx and
onnxruntime (the cv env):

    pixi run -e cv pytest tests/perception/test_onnx_models.py
"""

from __future__ import annotations  # names below exist only with torch

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

HAS_EXPORT = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "transformers", "onnx", "onnxscript", "onnxruntime")
)
needs_export = pytest.mark.skipif(
    not HAS_EXPORT, reason="needs torch, onnx and onnxruntime (the cv env)"
)
if HAS_EXPORT:
    import torch

    from minifsd.perception import depth_student as student
    from minifsd.perception import floor_dino, onnx_export, onnx_models
    from minifsd.train import depth_student as train_student
    from minifsd.train import dino as train_dino


def _photo(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)


@needs_export
def test_the_onnx_floor_model_gives_the_torch_models_logits(
    tmp_path: pathlib.Path,
) -> None:
    torch.manual_seed(0)
    model = train_dino.build_model(None).eval()  # tiny, random
    model.save(tmp_path / "floor")
    path = onnx_export.export_floor(tmp_path / "floor", tmp_path / "floor.onnx")
    runtime = onnx_models.OnnxFloor(path, threads=1)
    for seed in (0, 1):
        rgb = _photo(seed)
        x = torch.from_numpy(floor_dino.preprocess(rgb))
        with torch.inference_mode():
            expected = model(x)[0].numpy()
        np.testing.assert_allclose(
            runtime.logits(rgb), expected, rtol=1e-3, atol=1e-4
        )
        mask = runtime.predict(rgb)
        assert mask.shape == (240, 320) and mask.dtype == bool


@needs_export
def test_the_onnx_depth_model_gives_the_torch_models_disparity(
    tmp_path: pathlib.Path,
) -> None:
    torch.manual_seed(0)
    train_student.build_model(None).save_pretrained(tmp_path / "depth")
    path = onnx_export.export_depth(tmp_path / "depth", tmp_path / "depth.onnx")
    runtime = onnx_models.OnnxDepth(path, threads=1)
    reference = student.DepthStudent(tmp_path / "depth")
    for seed in (0, 1):
        rgb = _photo(seed)
        out = runtime.predict(rgb)
        assert out.shape == (240, 320) and out.dtype == np.float32
        np.testing.assert_allclose(
            out, reference.predict(rgb), rtol=1e-3, atol=1e-4
        )


def test_the_runtime_module_loads_without_torch() -> None:
    # The ROS env has no torch (M4.3), and the perception node imports
    # onnx_models there. A fresh interpreter shows what importing it pulls
    # in, whatever this test process has loaded already.
    code = (
        "import sys; import minifsd.perception.onnx_models; "
        "assert 'torch' not in sys.modules, 'imports torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
