"""Depth Anything V2 Small, used zero-shot (V5).

The model outputs relative disparity; minifsd.eval.depth_align turns it into
metres. Preprocessing is written here with OpenCV rather than taken from
transformers' image processor: the ONNX deployment (E6) needs the same steps
without transformers, and one copy of them keeps evaluation and deployment in
step. It follows the original Depth Anything V2 code: resize so the short
side is at least 518 px with both sides multiples of 14 (the ViT patch size),
bicubic, then ImageNet normalisation.

torch and transformers are imported inside DepthAnything, so this module
(and preprocess) also loads in the core env, which has neither.
"""

import math
import pathlib

import cv2
import numpy as np

SHORT_SIDE = 518  # the training resolution
PATCH = 14  # ViT-S/14: both sides must be multiples of this
MEAN = (0.485, 0.456, 0.406)  # ImageNet, RGB
STD = (0.229, 0.224, 0.225)


def _multiple_at_least(x: float, minimum: int) -> int:
    # The nearest multiple of PATCH, but never below minimum.
    y = round(x / PATCH) * PATCH
    return y if y >= minimum else math.ceil(x / PATCH) * PATCH


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Turns an RGB image into the model's input tensor.

    Args:
        rgb: (H, W, 3) uint8.

    Returns:
        (1, 3, h, w) float32, normalised, with min(h, w) >= SHORT_SIDE and
        both sides multiples of 14.

    Raises:
        ValueError: If the image isn't (H, W, 3).
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) RGB, got {rgb.shape}")
    h, w = rgb.shape[:2]
    scale = SHORT_SIDE / min(h, w)
    min_h = SHORT_SIDE if h <= w else 0
    min_w = SHORT_SIDE if w < h else 0
    new_h = _multiple_at_least(h * scale, min_h)
    new_w = _multiple_at_least(w * scale, min_w)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    mean = np.array(MEAN, np.float32)
    std = np.array(STD, np.float32)
    x = (resized.astype(np.float32) / np.float32(255.0) - mean) / std
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


class DepthAnything:
    """The model, loaded from a local folder (no network at run time)."""

    def __init__(self, model_dir: pathlib.Path, device: str = "cpu") -> None:
        """Loads the weights.

        Args:
            model_dir: Folder with config.json and model.safetensors, e.g.
                data/models/depth-anything-v2-small.
            device: "cpu" or "mps" (the Apple GPU).
        """
        import torch
        import transformers

        self._torch = torch
        self.device = device
        self.model = (
            transformers.AutoModelForDepthEstimation.from_pretrained(
                str(model_dir)
            )
            .eval()
            .to(device)
        )

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Relative disparity at the input image's size.

        Args:
            rgb: (H, W, 3) uint8.

        Returns:
            (H, W) float32. Larger means closer; the scale is arbitrary until
            depth_align fits it.
        """
        torch = self._torch
        h, w = rgb.shape[:2]
        x = torch.from_numpy(preprocess(rgb)).to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=x).predicted_depth  # (1, h', w')
            out = torch.nn.functional.interpolate(
                out[:, None], size=(h, w), mode="bilinear", align_corners=False
            )
        return np.asarray(out[0, 0].cpu().numpy(), dtype=np.float32)
