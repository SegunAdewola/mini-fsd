"""The demo script: a folder of photos in, overlays out (V8).

    python -m minifsd.demo.run --input photos/ --output runs/demo/

Each photo is shrunk to WORKING_SIZE on its long side for the models, and
the results are drawn on the full-size photo. The classical floor baseline's
blur, edge thresholds and fill tolerance are in pixels and were tuned on
640 x 480 NYUv2 images, so it has to see photos at about that scale; a
12-megapixel phone photo would also take far too long. OpenCV applies the
EXIF rotation when it reads a photo, so portrait shots come out upright.

For now it draws the classical floor mask. The trained models replace it in
M3, and M6 turns this into the offline ONNX demo.
"""

import argparse
import pathlib
import time
from collections.abc import Callable, Sequence

import cv2
import numpy as np

from minifsd.demo import overlay
from minifsd.perception import floor_classical

WORKING_SIZE = 640  # long side, in pixels, that the models see
EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_images(folder: pathlib.Path) -> list[pathlib.Path]:
    """The folder's images (png, jpg, jpeg in any case), sorted by name."""
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )


def process(
    rgb: np.ndarray,
    floor: Callable[[np.ndarray], np.ndarray] = floor_classical.floor_mask,
) -> np.ndarray:
    """Runs the models on one photo and draws their outputs on it.

    Args:
        rgb: (H, W, 3) uint8, any size.
        floor: RGB in, (h, w) bool floor mask out.

    Returns:
        (H, W, 3) uint8: the photo with the overlays, at its own size.
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, WORKING_SIZE / max(h, w))
    small = rgb
    if scale < 1.0:
        size = (round(w * scale), round(h * scale))  # OpenCV takes (w, h)
        small = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    mask = floor(small)
    # Nearest neighbour keeps the mask boolean: no half-floor pixels.
    full = cv2.resize(
        mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return overlay.draw_floor(rgb, full.astype(bool))


def main(argv: Sequence[str] | None = None) -> None:
    """Writes <name>_overlay.png for every photo in the input folder."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    paths = find_images(args.input)
    if not paths:
        raise SystemExit(f"no {'/'.join(EXTENSIONS)} images in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skipped {path.name}: not a readable image")
            continue
        start = time.perf_counter()
        result = process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ms = (time.perf_counter() - start) * 1000
        out = args.output / f"{path.stem}_overlay.png"
        cv2.imwrite(str(out), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"  {path.name}: {ms:.0f} ms -> {out}")
    print(f"{len(paths)} photos, overlays in {args.output}/")


if __name__ == "__main__":
    main()
