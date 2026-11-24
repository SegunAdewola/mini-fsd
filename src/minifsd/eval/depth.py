"""Scores a relative depth model on NYUv2, with both kinds of alignment.

    pixi run -e cv python -m minifsd.eval.depth --split test --device mps

Each image is scored twice (see minifsd.eval.depth_align):
- per-image fit: scale and shift fitted on that image's ground truth, the
  standard zero-shot protocol and an upper bound;
- global fit: one scale and shift fitted on NYUv2 train images, applied to
  every image unchanged, which is what a deployed robot could do.

AbsRel and delta1 are averaged over images, as in the Depth Anything papers.
Latency is the model call alone (preprocessing included), not the fits.
"""

import argparse
import dataclasses
import pathlib
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

import numpy as np

from minifsd.eval import depth_align, metrics


class DepthSample(Protocol):
    """What evaluate_depth needs from a sample (nyuv2.Sample fits)."""

    @property
    def index(self) -> int: ...  # noqa: D102
    @property
    def rgb(self) -> np.ndarray: ...  # noqa: D102
    @property
    def depth(self) -> np.ndarray: ...  # noqa: D102


@dataclasses.dataclass(frozen=True)
class Scores:
    """Mean AbsRel (lower is better) and delta1 (higher is better)."""

    abs_rel: float
    delta1: float


@dataclasses.dataclass(frozen=True)
class DepthResult:
    """Scores under both alignments, plus latency."""

    n_images: int
    per_image_fit: Scores
    global_fit: Scores
    p50_ms: float
    p95_ms: float
    # (index, AbsRel with the per-image fit), in input order.
    per_image: list[tuple[int, float]]


def evaluate_depth(
    samples: Iterable[DepthSample],
    predict: Callable[[np.ndarray], np.ndarray],
    global_fit: depth_align.Fit,
) -> DepthResult:
    """Runs the model on every sample and scores it both ways.

    Args:
        samples: Images with ground-truth depth in metres.
        predict: RGB (H, W, 3) to relative disparity (H, W).
        global_fit: The scale and shift fitted on training images.

    Returns:
        The scores.

    Raises:
        ValueError: If there are no samples.
    """
    local: list[tuple[float, float]] = []
    fixed: list[tuple[float, float]] = []
    times: list[float] = []
    per_image: list[tuple[int, float]] = []
    for s in samples:
        start = time.perf_counter()
        disparity = predict(s.rgb)
        times.append((time.perf_counter() - start) * 1000)

        own = depth_align.fit(disparity, s.depth).to_depth(disparity)
        shared = global_fit.to_depth(disparity)
        local.append(
            (metrics.abs_rel(own, s.depth), metrics.delta1(own, s.depth))
        )
        fixed.append(
            (metrics.abs_rel(shared, s.depth), metrics.delta1(shared, s.depth))
        )
        per_image.append((s.index, local[-1][0]))
    if not times:
        raise ValueError("no samples")

    def mean(rows: list[tuple[float, float]]) -> Scores:
        a = np.array(rows)
        return Scores(float(a[:, 0].mean()), float(a[:, 1].mean()))

    return DepthResult(
        n_images=len(times),
        per_image_fit=mean(local),
        global_fit=mean(fixed),
        p50_ms=float(np.percentile(times, 50)),
        p95_ms=float(np.percentile(times, 95)),
        per_image=per_image,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Fits the global alignment on train, then scores the chosen split."""
    # Imported here so the scoring code above needs neither h5py nor torch.
    from minifsd.data import nyuv2
    from minifsd.perception import depth_anything

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--data", default="data/nyuv2", type=pathlib.Path)
    parser.add_argument(
        "--model",
        default="data/models/depth-anything-v2-small",
        type=pathlib.Path,
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument(
        "--fit-images",
        type=int,
        default=100,
        help="train images for the global fit (a fixed, seeded sample)",
    )
    parser.add_argument("--limit", type=int, help="score only the first N")
    args = parser.parse_args(argv)

    model = depth_anything.DepthAnything(args.model, device=args.device)
    splits = nyuv2.load_split(args.data / "splits.mat")
    rng = np.random.default_rng(0)
    fit_indices = sorted(
        rng.choice(splits["train"], size=args.fit_images, replace=False)
    )
    indices = splits[args.split][: args.limit]

    with nyuv2.NyuV2(args.data / "nyu_depth_v2_labeled.mat") as data:
        pairs = []
        for i in fit_indices:
            s = nyuv2.crop(data[int(i)])
            pairs.append((model.predict(s.rgb), s.depth))
        global_fit = depth_align.fit_many(pairs)
        del pairs
        model.predict(data[0].rgb)  # warm-up, so latency excludes start-up
        samples = [nyuv2.crop(data[i]) for i in indices]
    result = evaluate_depth(samples, model.predict, global_fit)

    print(f"Depth Anything V2 Small, zero-shot, NYUv2 {args.split}, Eigen crop")
    print(f"  images                  {result.n_images}")
    print(
        f"  global fit ({args.fit_images} train images): "
        f"scale {global_fit.scale:.4g}, shift {global_fit.shift:.4g}"
    )
    for name, sc in (
        ("per-image fit (oracle)", result.per_image_fit),
        ("global fit", result.global_fit),
    ):
        print(f"  {name:24s} AbsRel {sc.abs_rel:.3f}  delta1 {sc.delta1:.3f}")
    print(
        f"  latency p50 / p95       {result.p50_ms:.0f} / "
        f"{result.p95_ms:.0f} ms (torch, {args.device})"
    )


if __name__ == "__main__":
    main()
