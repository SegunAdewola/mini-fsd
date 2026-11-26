"""The floor and depth models through ONNX Runtime, without torch (M4.4).

The ROS env has ONNX Runtime but no torch (M4.3), so the perception node
runs the picked models from files that minifsd.perception.onnx_export wrote
in the cv env. Each class has the same predict() as its torch twin
(floor_dino.FloorDino, depth_student.DepthStudent), and the M4.4 tests check
that the outputs agree.

onnxruntime is imported inside the session helper, so this module loads in
any env, and never imports torch.
"""

import pathlib
from typing import Any

import cv2
import numpy as np

from minifsd.perception import depth_student, floor_segformer


def session(path: pathlib.Path, threads: int) -> Any:
    """An ONNX Runtime CPU session capped at `threads` (the edge budget)."""
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return onnxruntime.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )


class OnnxFloor:
    """The DINOv2 floor model (E2b) from its ONNX export."""

    def __init__(self, path: pathlib.Path, threads: int = 2) -> None:
        """Loads the model.

        Args:
            path: An .onnx file from onnx_export.export_floor.
            threads: ONNX Runtime's thread cap.
        """
        self._session = session(path, threads)
        node = self._session.get_inputs()[0]
        self._input = node.name
        self._height, self._width = int(node.shape[2]), int(node.shape[3])

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        # floor_dino.preprocess, at the size the graph was exported for. It
        # can't be imported here (floor_dino needs torch); the M4.4 test
        # holds the two to the same logits.
        h, w = rgb.shape[:2]
        shrink = h > self._height or w > self._width
        resized = cv2.resize(
            rgb,
            (self._width, self._height),
            interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
        )
        return floor_segformer.normalise(resized)[None]

    def logits(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB to (2, h, w) logits on the patch grid."""
        out = self._session.run(None, {self._input: self._preprocess(rgb)})
        return np.asarray(out[0][0], np.float32)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB to an (H, W) bool floor mask."""
        h, w = rgb.shape[:2]
        return floor_segformer.mask_from_logits(self.logits(rgb), h, w)


class OnnxDepth:
    """The distilled depth student (E3b) from its ONNX export."""

    def __init__(self, path: pathlib.Path, threads: int = 2) -> None:
        """Loads the model.

        Args:
            path: An .onnx file from onnx_export.export_depth.
            threads: ONNX Runtime's thread cap.
        """
        self._session = session(path, threads)
        self._input = self._session.get_inputs()[0].name

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 RGB to (H, W) float32 relative disparity."""
        x = depth_student.preprocess(rgb)
        small = np.asarray(
            self._session.run(None, {self._input: x})[0][0, 0], np.float32
        )
        h, w = rgb.shape[:2]
        big: np.ndarray = cv2.resize(
            small, (w, h), interpolation=cv2.INTER_LINEAR
        )
        return big.astype(np.float32)
