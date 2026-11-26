"""Models run elsewhere, scored here with the same metrics (E3a).

    pixi run python -m minifsd.eval.precomputed --out data/da3-inputs
        --datasets nyuv2-test nyuv2-fit nyuv2-val sim:data/sim/v9/val

(one command, wrapped here to fit)

Depth Anything 3's package pins numpy<2 and loads 3D and video tools on
import, so it can't share our environments (decided 2026-11-25: run it on
Colab only). The trip:

1. export: write every image the harness would hand a depth model, exactly
   as it would (NYUv2's Eigen crop included), named by a hash of its pixels.
2. On Colab, minifsd.train.da3_predict writes one depth map per image into
   an .npz keyed the same way.
3. PrecomputedDepth looks each image's prediction up by the same hash, so
   the harness scores it with the same fit and metrics as any other depth
   model. Its candidate is untimed: latency on Colab's GPU says nothing
   about the M1, so none is logged.
"""

import argparse
import hashlib
import os
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import cv2
import numpy as np

EPS = 1e-6  # floor for depth before 1/depth


def key(rgb: np.ndarray) -> str:
    """A name for an image from its shape and pixels."""
    digest = hashlib.sha1(str(rgb.shape).encode())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def export_images(samples: Iterable[Any], out_dir: pathlib.Path) -> int:
    """Writes each sample's RGB as <key>.png (lossless); returns how many.

    Images already written (by this call or an earlier one) are skipped, so
    overlapping datasets cost nothing extra.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    seen = set()
    for sample in samples:
        name = key(sample.rgb)
        if name in seen:
            continue
        seen.add(name)
        path = out_dir / f"{name}.png"
        if not path.exists():
            cv2.imwrite(str(path), sample.rgb[..., ::-1])
            written += 1
    return written


def save(path: pathlib.Path, maps: Mapping[str, np.ndarray]) -> None:
    """Writes {key: depth} as a compressed .npz, whole or not at all.

    Written under a temporary name and renamed when complete, so a dropped
    session never leaves a file that looks whole.
    """
    arrays: dict[str, Any] = dict(maps)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as f:  # a file object, so numpy adds no suffix
        np.savez_compressed(f, **arrays)
    os.replace(part, path)


def read_rgb(path: pathlib.Path) -> np.ndarray:
    """An exported image back as (H, W, 3) uint8 RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"can't read {path}")
    return np.ascontiguousarray(bgr[..., ::-1])


class PrecomputedDepth:
    """Predictions from an .npz of depth maps, as relative disparity."""

    def __init__(self, path: pathlib.Path) -> None:
        """Opens the predictions; each is read when first asked for.

        Args:
            path: An .npz of (H, W) depth maps keyed by key(rgb).
        """
        self._data = np.load(path)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """1 / depth for this image, at its size.

        Raises:
            KeyError: If the image wasn't predicted.
        """
        name = key(rgb)
        if name not in self._data.files:
            raise KeyError(f"no prediction for image {name}")
        depth = self._data[name].astype(np.float32)
        h, w = rgb.shape[:2]
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        disparity: np.ndarray = 1.0 / np.maximum(depth, EPS)
        return disparity


def main(argv: Sequence[str] | None = None) -> None:
    """Exports the images of the given harness datasets."""
    from minifsd.eval import harness

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)
    for name in args.datasets:
        n = export_images(harness.load_dataset(name), args.out)
        print(f"{name}: {n} new images")
    print(f"{len(list(args.out.glob('*.png')))} images in {args.out}/")


if __name__ == "__main__":
    main()
