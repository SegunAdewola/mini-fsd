"""E3b's small depth network at run time: SegFormer-B0 at 160 x 120.

One output channel, relative disparity: larger is closer, and the scale is
fitted later, as for the zero-shot teachers (minifsd.eval.depth_align). The
input is a quarter of E2a's pixels, to aim at the 50 ms budget; depth for
the costmap doesn't need the full image. Preprocessing is floor_segformer's
normalisation at this smaller size, so the ONNX path (E6) can share it.

torch and transformers are imported inside DepthStudent, so the rest of
this module loads in the core env.
"""

import pathlib

import cv2
import numpy as np

from minifsd.perception import floor_segformer

INPUT_HEIGHT = 120
INPUT_WIDTH = 160


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Any RGB photo to the student's (1, 3, 120, 160) float32 input."""
    resized = cv2.resize(
        rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA
    )
    return floor_segformer.normalise(resized)[None]


class DepthStudent:
    """A trained student (a save_pretrained folder) as a depth predictor."""

    def __init__(self, path: pathlib.Path, device: str = "cpu") -> None:
        """Loads the model.

        Args:
            path: Folder written by save_pretrained, e.g. a run's best/.
            device: "cpu", "mps" or "cuda".
        """
        import torch
        from transformers import SegformerForSemanticSegmentation

        self._torch = torch
        self._device = device
        self._model = SegformerForSemanticSegmentation.from_pretrained(path)
        self._model.to(device).eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB to (H, W) float32 relative disparity."""
        x = self._torch.from_numpy(preprocess(rgb)).to(self._device)
        with self._torch.inference_mode():
            out = self._model(pixel_values=x).logits[0, 0]
        h, w = rgb.shape[:2]
        small = out.float().cpu().numpy()
        big: np.ndarray = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_LINEAR
        )
        return big.astype(np.float32)
