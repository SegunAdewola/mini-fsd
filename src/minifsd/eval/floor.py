"""Scores a floor segmenter on a dataset: IoU and latency.

Run the classical baseline on NYUv2 (V3):

    pixi run python -m minifsd.eval.floor --split train   # for tuning
    pixi run python -m minifsd.eval.floor --split test    # once, to report

It prints one table and saves the worst images as side-by-side PNGs
(image | prediction | ground truth) for failure analysis.
"""

import argparse
import dataclasses
import pathlib
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np

from minifsd.eval import metrics

# Rows above the bottom edge where "bottom centre" is checked in the ground
# truth. Matches floor_classical's default seed.
BOTTOM_ROWS_UP = 5


class FloorSample(Protocol):
    """What evaluate_floor needs from a sample. nyuv2.Sample fits.

    Read-only properties, so frozen dataclasses match the protocol too.
    """

    @property
    def index(self) -> int: ...  # noqa: D102
    @property
    def rgb(self) -> np.ndarray: ...  # noqa: D102
    @property
    def floor(self) -> np.ndarray: ...  # noqa: D102
    @property
    def labeled(self) -> np.ndarray: ...  # noqa: D102


@dataclasses.dataclass(frozen=True)
class FloorResult:
    """Scores for one model on one split.

    Attributes:
        n_images: Images scored.
        iou: Micro IoU, total intersection over total union.
        n_bottom_floor: Images whose ground truth is floor at the bottom
            centre: the case a forward robot camera is in.
        iou_bottom_floor: Micro IoU over just those images.
        precision: Of the pixels called floor, the share that are floor,
            summed over the dataset; NaN if nothing was called floor. Low
            precision is the dangerous error: the robot plans over a table.
        recall: Of the floor pixels, the share found; NaN if there were
            none.
        p50_ms: Median time per image, milliseconds.
        p95_ms: 95th percentile time per image, milliseconds.
        per_image: (sample index, IoU) for every image, for finding failures.
    """

    n_images: int
    iou: float
    n_bottom_floor: int
    iou_bottom_floor: float
    precision: float
    recall: float
    p50_ms: float
    p95_ms: float
    per_image: tuple[tuple[int, float], ...]

    def worst(self, k: int) -> list[int]:
        """Sample indices of the k lowest-IoU images, worst first."""
        ranked = sorted(self.per_image, key=lambda pair: pair[1])
        return [index for index, _ in ranked[:k]]


def evaluate_floor(
    samples: Iterable[FloorSample],
    predict: Callable[[np.ndarray], np.ndarray],
) -> FloorResult:
    """Runs predict on every sample and scores it against the ground truth.

    Only labeled pixels count (metrics' `valid` argument). Latency covers the
    predict call alone, not loading.

    Args:
        samples: Images with floor and labeled masks.
        predict: Maps an (H, W, 3) uint8 RGB image to an (H, W) bool mask.

    Returns:
        The scores.

    Raises:
        ValueError: If there are no samples.
    """
    inter = union = inter_b = union_b = n_bottom = 0
    called = actual = 0  # predicted floor pixels; true floor pixels
    times: list[float] = []
    per_image: list[tuple[int, float]] = []
    for s in samples:
        start = time.perf_counter()
        pred = predict(s.rgb)
        times.append(time.perf_counter() - start)
        i, u = metrics.intersection_union(pred, s.floor, s.labeled)
        inter, union = inter + i, union + u
        called += int(np.count_nonzero(pred & s.labeled))
        actual += int(np.count_nonzero(s.floor & s.labeled))
        per_image.append((s.index, 1.0 if u == 0 else i / u))
        h, w = s.floor.shape
        if s.floor[h - 1 - BOTTOM_ROWS_UP, w // 2]:
            n_bottom += 1
            inter_b, union_b = inter_b + i, union_b + u
    if not times:
        raise ValueError("no samples to evaluate")
    ms = np.array(times) * 1000
    return FloorResult(
        n_images=len(times),
        iou=inter / union if union else 1.0,
        n_bottom_floor=n_bottom,
        iou_bottom_floor=inter_b / union_b if union_b else float("nan"),
        precision=inter / called if called else float("nan"),
        recall=inter / actual if actual else float("nan"),
        p50_ms=float(np.percentile(ms, 50)),
        p95_ms=float(np.percentile(ms, 95)),
        per_image=tuple(per_image),
    )


def _side_by_side(
    rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray
) -> np.ndarray:
    """Image | prediction in green | ground truth in blue, as RGB uint8."""

    def tint(mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
        out = rgb.astype(np.float32)
        out[mask] = 0.5 * out[mask] + 0.5 * np.array(colour, np.float32)
        return out.astype(np.uint8)

    return np.hstack([rgb, tint(pred, (0, 255, 0)), tint(gt, (0, 0, 255))])


def main(argv: Sequence[str] | None = None) -> None:
    """Scores the classical baseline on NYUv2 and saves the worst images."""
    # Imported here so the scoring code above needs neither OpenCV nor h5py.
    import cv2

    from minifsd.data import nyuv2
    from minifsd.perception import floor_classical

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--data", default="data/nyuv2", type=pathlib.Path)
    parser.add_argument("--out", default="runs/v3-floor-classical")
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args(argv)

    indices = nyuv2.load_split(args.data / "splits.mat")[args.split]
    with nyuv2.NyuV2(args.data / "nyu_depth_v2_labeled.mat") as data:
        samples = [nyuv2.crop(data[i]) for i in indices]
    result = evaluate_floor(samples, floor_classical.floor_mask)

    print(f"Classical floor baseline, NYUv2 {args.split}, Eigen crop")
    print(f"  images                 {result.n_images}")
    print(f"  floor IoU (all)        {result.iou:.3f}")
    print(
        f"  floor IoU (bottom is floor, {result.n_bottom_floor} images) "
        f"{result.iou_bottom_floor:.3f}"
    )
    print(
        f"  latency p50 / p95      {result.p50_ms:.1f} / {result.p95_ms:.1f} ms"
    )

    out = pathlib.Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    by_index = {s.index: s for s in samples}
    scores = dict(result.per_image)
    for rank, index in enumerate(result.worst(args.worst), start=1):
        s = by_index[index]
        panel = _side_by_side(s.rgb, floor_classical.floor_mask(s.rgb), s.floor)
        name = f"worst{rank}_nyu{index}_iou{scores[index]:.2f}.png"
        cv2.imwrite(str(out / name), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"  worst {args.worst} saved to {out}/")


if __name__ == "__main__":
    main()
