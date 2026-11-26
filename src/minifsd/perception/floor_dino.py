"""Frozen DINOv2 ViT-S/14 features with a light floor head (E2b).

E2b asks whether a foundation model's features, left as they are, carry
across the sim-to-real gap better than a network fine-tuned end to end on
sim (E2a). So the backbone never trains; only the head does.

The head is the DINOv2 paper's linear segmentation probe, multi-layer
version: the patch tokens of the last four blocks, each through the
backbone's final layer norm, concatenated (4 x 384 channels), then batch
norm and a 1 x 1 convolution to two classes. Logits come out on the 14 px
patch grid and are scaled up to the photo's size, as for SegFormer.

DINOv3 ViT-S/16 was the plan's first choice; its weights are gated, so
DINOv2 (open) stands in, and the report says so.

Unlike floor_segformer this module imports torch at load: the model class
needs it. The harness imports it only when this candidate is asked for.
"""

import json
import pathlib

import cv2
import numpy as np
import torch
from torch import nn
from transformers import Dinov2Model

from minifsd.perception import floor_segformer

BACKBONE = "facebook/dinov2-small"
PATCH = 14
# The multiples of 14 nearest the camera's 240 x 320: a 17 x 23 patch grid.
INPUT_HEIGHT = 238
INPUT_WIDTH = 322
LAYERS = 4  # last blocks whose tokens feed the head
HEAD_FILE = "head.pt"
SETTINGS_FILE = "head.json"
BACKBONE_DIR = "backbone"


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Any RGB photo to the model's (1, 3, 238, 322) float32 input."""
    h, w = rgb.shape[:2]
    shrink = h > INPUT_HEIGHT or w > INPUT_WIDTH
    resized = cv2.resize(
        rgb,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
    )
    return floor_segformer.normalise(resized)[None]


class DinoFloor(nn.Module):
    """A frozen DINOv2 backbone and a trainable linear head."""

    def __init__(self, backbone: Dinov2Model, layers: int = LAYERS) -> None:
        """Wraps the backbone and freezes it.

        Args:
            backbone: A DINOv2 model from transformers.
            layers: How many of the last blocks feed the head.
        """
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.layers = layers
        channels = backbone.config.hidden_size * layers
        self.norm = nn.BatchNorm2d(channels)
        self.classifier = nn.Conv2d(channels, 2, kernel_size=1)

    def train(self, mode: bool = True) -> "DinoFloor":
        """Train mode for the head only; the backbone stays in eval."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) input to (B, 2, H / 14, W / 14) logits."""
        b, _, h, w = pixel_values.shape
        with torch.no_grad():
            out = self.backbone(
                pixel_values=pixel_values, output_hidden_states=True
            )
            tokens = [
                self.backbone.layernorm(hidden[:, 1:])  # drop the CLS token
                for hidden in out.hidden_states[-self.layers :]
            ]
        x = torch.cat(tokens, dim=-1)  # (B, patches, channels)
        x = x.transpose(1, 2).reshape(b, -1, h // PATCH, w // PATCH)
        logits: torch.Tensor = self.classifier(self.norm(x))
        return logits

    def save(self, folder: pathlib.Path) -> None:
        """Writes the backbone and head, so loading needs no download."""
        folder.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(folder / BACKBONE_DIR)
        head = {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("backbone.")
        }
        torch.save(head, folder / HEAD_FILE)
        (folder / SETTINGS_FILE).write_text(
            json.dumps({"layers": self.layers}) + "\n"
        )

    @classmethod
    def load(cls, folder: pathlib.Path) -> "DinoFloor":
        """Reads a folder written by save()."""
        settings = json.loads((folder / SETTINGS_FILE).read_text())
        backbone = Dinov2Model.from_pretrained(folder / BACKBONE_DIR)
        model = cls(backbone, layers=settings["layers"])
        head = torch.load(folder / HEAD_FILE, weights_only=True)
        model.load_state_dict(head, strict=False)
        return model


class FloorDino:
    """A trained DinoFloor folder as a floor masker, for the harness."""

    def __init__(self, path: pathlib.Path, device: str = "cpu") -> None:
        """Loads the model.

        Args:
            path: Folder written by DinoFloor.save, e.g. a run's best/.
            device: "cpu", "mps" or "cuda".
        """
        self._device = device
        self._model = DinoFloor.load(path).to(device).eval()

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB to an (H, W) bool floor mask."""
        x = torch.from_numpy(preprocess(rgb)).to(self._device)
        with torch.inference_mode():
            logits = self._model(x)[0]
        h, w = rgb.shape[:2]
        return floor_segformer.mask_from_logits(
            logits.float().cpu().numpy(), h, w
        )
