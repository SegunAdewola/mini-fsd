"""Depth Anything 3 Small, zero-shot, on Colab (E3a).

    /content/da3env/bin/python -m minifsd.train.da3_predict
        --images /content/da3-inputs --out /content/drive/MyDrive/da3.npz

(one command, wrapped here to fit)

DA3's package pins numpy<2 and loads 3D and video tools on import, so it
lives in its own virtual environment on Colab, never in ours (decided
2026-11-25). This module runs inside that environment: it reads the images
that minifsd.eval.precomputed exported, runs DA3-Small on each (one view,
no camera given), and saves the depth maps, resized to each image's size,
in one .npz keyed by file name. The harness then scores them on the M1
like any other depth model, with no latency (it ran on a GPU elsewhere).

It imports only numpy, OpenCV and minifsd.eval.precomputed, which work
with DA3's older numpy.
"""

import argparse
import pathlib
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from minifsd.eval import precomputed

WEIGHTS = "depth-anything/DA3-SMALL"  # Apache-2.0 on the Hugging Face Hub

DepthFn = Callable[[np.ndarray], np.ndarray]  # (H, W, 3) RGB -> depth


def predict_folder(
    images: pathlib.Path, out: pathlib.Path, depth_of: DepthFn
) -> int:
    """Runs depth_of on every <key>.png; saves {key: depth} as float16.

    Saved with precomputed.save, so a dropped session leaves no half file.

    Returns:
        How many images were predicted.
    """
    maps = {}
    for path in sorted(images.glob("*.png")):
        rgb = precomputed.read_rgb(path)
        depth = np.asarray(depth_of(rgb), np.float32)
        h, w = rgb.shape[:2]
        if depth.shape != (h, w):
            depth = np.asarray(
                cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            )
        maps[path.stem] = depth.astype(np.float16)
    precomputed.save(out, maps)
    return len(maps)


def da3_small(device: str = "cuda") -> DepthFn:
    """DA3-Small as an RGB to depth function (imports DA3; Colab only)."""
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(WEIGHTS).to(device).eval()

    def depth_of(rgb: np.ndarray) -> np.ndarray:
        prediction = model.inference([rgb])
        return np.asarray(prediction.depth[0])

    return depth_of


def main(argv: Sequence[str] | None = None) -> None:
    """Predicts a folder of exported images."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    n = predict_folder(args.images, args.out, da3_small())
    print(f"{n} depth maps in {args.out}")


if __name__ == "__main__":
    main()
