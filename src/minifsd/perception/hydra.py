"""HydraNet-lite: one frozen DINOv2-S, three heads (H3, E4).

E4 asks the design's "Tesla question": do separate specialist models beat
one shared backbone with a head per task, and how much latency does sharing
save? The backbone is E2's winner, DINOv2-S, frozen, run once per image.

    patch tokens of the last 4 blocks, concatenated: (B, 1536, 17, 23)
    |- floor: batch norm, 1 x 1 conv -> 2 classes            (E2b's probe)
    '- neck: batch norm, 1 x 1 conv to 256, upsample x2 to stride 7 (34 x 46),
       3 x 3 conv
       |- depth: 1 x 1 conv -> log depth
       '- boxes: CenterNet heads (minifsd.perception.centernet), each a
          3 x 3 conv to 64 and a 1 x 1 conv: heatmap (one per class), box
          size, centre offset

Input and preprocessing are floor_dino's (238 x 322, ImageNet-normalised).
"""

import dataclasses
import json
import math
import pathlib
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional
from transformers import Dinov2Model

from minifsd.data import boxes as boxes_lib
from minifsd.eval import detection
from minifsd.perception import centernet, floor_dino, floor_segformer

NECK = 256
BRANCH = 64
# CenterNet's heatmap bias: sigmoid(-2.19) is 0.1, so no cell starts out
# confident and the focal loss stays sane on the first steps.
HEATMAP_BIAS = -math.log((1 - 0.1) / 0.1)
HEADS_FILE = "heads.pt"
SETTINGS_FILE = "hydra.json"


TASKS = ("floor", "depth", "detect")


@dataclasses.dataclass(frozen=True)
class Heads:
    """One forward pass's outputs; a head not asked for is None."""

    floor: torch.Tensor  # (B, 2, 17, 23) logits
    log_depth: torch.Tensor  # (B, 1, 34, 46)
    heatmap: torch.Tensor  # (B, C, 34, 46) logits
    size: torch.Tensor  # (B, 2, 34, 46) box width, height in cells
    offset: torch.Tensor  # (B, 2, 34, 46) centre within its cell


def _branch(neck: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(neck, BRANCH, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(BRANCH, out, 1),
    )


class HydraNet(nn.Module):
    """A frozen DINOv2 backbone with floor, depth and box heads."""

    def __init__(
        self,
        backbone: Dinov2Model,
        n_classes: int = len(boxes_lib.CATEGORIES),
        neck: int = NECK,
        layers: int = floor_dino.LAYERS,
    ) -> None:
        """Wraps the backbone, freezes it, and adds fresh heads.

        Args:
            backbone: A DINOv2 model from transformers.
            n_classes: Detection classes.
            neck: Channels of the shared dense neck.
            layers: How many of the last blocks feed the heads.
        """
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.settings = {"n_classes": n_classes, "neck": neck, "layers": layers}
        channels = backbone.config.hidden_size * layers
        self.floor_head = nn.Sequential(
            nn.BatchNorm2d(channels), nn.Conv2d(channels, 2, 1)
        )
        self.neck_in = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, neck, 1),
            nn.ReLU(inplace=True),
        )
        self.neck_out = nn.Sequential(
            nn.Conv2d(neck, neck, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.depth_head = nn.Conv2d(neck, 1, 1)
        self.heatmap_head = _branch(neck, n_classes)
        nn.init.constant_(self.heatmap_head[-1].bias, HEATMAP_BIAS)
        self.size_head = _branch(neck, 2)
        self.offset_head = _branch(neck, 2)

    def train(self, mode: bool = True) -> "HydraNet":
        """Train mode for the heads only; the backbone stays in eval."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self, pixel_values: torch.Tensor, heads: tuple[str, ...] = TASKS
    ) -> Heads:
        """(B, 3, 238, 322) input to the asked-for heads' outputs.

        Args:
            pixel_values: Preprocessed input.
            heads: Which of TASKS to run; the others come back as None, and
                the dense neck runs only for depth or detect. Training uses
                all three; a single-task candidate skips the rest, so its
                latency is its own.
        """
        b, _, h, w = pixel_values.shape
        layers = int(self.settings["layers"])
        with torch.no_grad():
            out = self.backbone(
                pixel_values=pixel_values, output_hidden_states=True
            )
            tokens = [
                self.backbone.layernorm(hidden[:, 1:])  # drop the CLS token
                for hidden in out.hidden_states[-layers:]
            ]
        x = torch.cat(tokens, dim=-1).transpose(1, 2)
        x = x.reshape(b, -1, h // floor_dino.PATCH, w // floor_dino.PATCH)
        none: Any = None
        floor = self.floor_head(x) if "floor" in heads else none
        log_depth = heatmap = size = offset = none
        if "depth" in heads or "detect" in heads:
            dense = functional.interpolate(
                self.neck_in(x),
                size=(h // centernet.STRIDE, w // centernet.STRIDE),
                mode="bilinear",
                align_corners=False,
            )
            dense = self.neck_out(dense)
            if "depth" in heads:
                log_depth = self.depth_head(dense)
            if "detect" in heads:
                heatmap = self.heatmap_head(dense)
                size = self.size_head(dense)
                offset = self.offset_head(dense)
        return Heads(floor, log_depth, heatmap, size, offset)

    def save(self, folder: pathlib.Path) -> None:
        """Writes backbone and heads, so loading needs no download."""
        folder.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(folder / floor_dino.BACKBONE_DIR)
        heads = {  # CPU copies: a model saved on a GPU must load anywhere
            k: v.detach().cpu()
            for k, v in self.state_dict().items()
            if not k.startswith("backbone.")
        }
        torch.save(heads, folder / HEADS_FILE)
        (folder / SETTINGS_FILE).write_text(json.dumps(self.settings) + "\n")

    @classmethod
    def load(cls, folder: pathlib.Path) -> "HydraNet":
        """Reads a folder written by save()."""
        settings = json.loads((folder / SETTINGS_FILE).read_text())
        backbone = Dinov2Model.from_pretrained(folder / floor_dino.BACKBONE_DIR)
        model = cls(backbone, **settings)
        heads = torch.load(
            folder / HEADS_FILE, map_location="cpu", weights_only=True
        )
        model.load_state_dict(heads, strict=False)
        return model


class Hydra:
    """A trained HydraNet as floor, depth and box predictors in photo pixels."""

    def __init__(self, path: pathlib.Path, device: str = "cpu") -> None:
        """Loads the model.

        Args:
            path: Folder written by HydraNet.save, e.g. a run's best/.
            device: "cpu", "mps" or "cuda".
        """
        self._device = device
        self._model = HydraNet.load(path).to(device).eval()

    def predict(
        self, rgb: np.ndarray, heads: tuple[str, ...]
    ) -> dict[str, Any]:
        """One forward pass, only the asked-for heads, at the photo's size.

        Returns:
            {"floor": (H, W) bool mask, "depth": (H, W) relative disparity
            (1 / depth), "detect": detections in the photo's pixels}, for
            the heads asked for.
        """
        x = torch.from_numpy(floor_dino.preprocess(rgb)).to(self._device)
        with torch.inference_mode():
            out = self._model(x, heads)
        h, w = rgb.shape[:2]
        answers: dict[str, Any] = {}
        if "floor" in heads:
            answers["floor"] = floor_segformer.mask_from_logits(
                out.floor[0].float().cpu().numpy(), h, w
            )
        if "depth" in heads:
            log_depth = out.log_depth[0, 0].float().cpu().numpy()
            depth = cv2.resize(
                np.exp(log_depth), (w, h), interpolation=cv2.INTER_LINEAR
            )
            answers["depth"] = 1.0 / np.maximum(depth, 1e-6)
        if "detect" in heads:
            answers["detect"] = self._boxes(out, w, h)
        return answers

    def predict_all(
        self, rgb: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[detection.Detection]]:
        """Every task from one forward pass: floor, disparity, boxes."""
        a = self.predict(rgb, TASKS)
        return a["floor"], a["depth"], a["detect"]

    def _boxes(self, out: Heads, w: int, h: int) -> list[detection.Detection]:
        found = centernet.decode(
            torch.sigmoid(out.heatmap[0]).float().cpu().numpy(),
            out.size[0].float().cpu().numpy(),
            out.offset[0].float().cpu().numpy(),
        )
        # The input was stretched to 238 x 322; undo that per axis.
        sx, sy = w / floor_dino.INPUT_WIDTH, h / floor_dino.INPUT_HEIGHT
        dets = []
        for f in found:
            left = min(max(f.x * sx, 0.0), w)
            top = min(max(f.y * sy, 0.0), h)
            right = min(max((f.x + f.w) * sx, 0.0), w)
            bottom = min(max((f.y + f.h) * sy, 0.0), h)
            dets.append(
                detection.Detection(
                    boxes_lib.CATEGORIES[f.cls],
                    left,
                    top,
                    right - left,
                    bottom - top,
                    f.score,
                )
            )
        return dets
