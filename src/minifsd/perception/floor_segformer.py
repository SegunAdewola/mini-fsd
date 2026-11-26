"""SegFormer-B0 floor segmentation, trained on v9 (E2a).

Two classes: 0 not floor, 1 floor. The model sees every image at the sim
camera's size, 320 x 240, as it was trained; its logits come out at a
quarter of that and are scaled back to the photo's own size here.

Pre- and postprocessing are numpy and OpenCV, as in depth_anything, so the
ONNX deployment (E6) reuses them without transformers. torch and
transformers are imported inside FloorSegformer, so the rest of this module
loads in the core env.
"""

import pathlib

import cv2
import numpy as np

INPUT_HEIGHT = 240
INPUT_WIDTH = 320
MEAN = (0.485, 0.456, 0.406)  # ImageNet, RGB: the encoder's pretraining
STD = (0.229, 0.224, 0.225)
FLOOR = 1  # class index


def normalise(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB to (3, H, W) float32, ImageNet-normalised."""
    scaled = rgb.astype(np.float32) / np.float32(255.0)
    x = (scaled - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Any RGB photo to the model's (1, 3, 240, 320) float32 input.

    Real photos are 4:3 like the sim camera, so this is a plain resize.
    INTER_AREA when shrinking avoids aliasing on 12 MP photos.
    """
    h, w = rgb.shape[:2]
    shrink = h > INPUT_HEIGHT or w > INPUT_WIDTH
    resized = cv2.resize(
        rgb,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
    )
    return normalise(resized)[None]


def mask_from_logits(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    """(2, h, w) logits to a (height, width) bool floor mask.

    The floor-minus-other margin is resized, not the argmax, so the edge
    falls between pixels smoothly instead of in quarter-size steps.
    """
    margin = (logits[FLOOR] - logits[1 - FLOOR]).astype(np.float32)
    big = cv2.resize(margin, (width, height), interpolation=cv2.INTER_LINEAR)
    mask: np.ndarray = big > 0
    return mask


class FloorSegformer:
    """A trained checkpoint (a save_pretrained folder) as a floor masker."""

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
        """(H, W, 3) uint8 RGB to an (H, W) bool floor mask."""
        x = self._torch.from_numpy(preprocess(rgb)).to(self._device)
        with self._torch.inference_mode():
            logits = self._model(pixel_values=x).logits[0]
        h, w = rgb.shape[:2]
        return mask_from_logits(logits.float().cpu().numpy(), h, w)
