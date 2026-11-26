"""Makes the report's E2 failure figure: floor overlays on nyuv2-val.

    pixi run -e cv python tools/report_e2_overlays.py

Rows are three nyuv2-val images where v9-trained SegFormer called raised
surfaces floor: a kitchen counter, a bed, and shelves over a desk in a room
with no visible floor (picked while looking at E2a's failures). Columns:
the photo, then three models. Blue is floor found, orange is floor called
where there is none, and missed floor is left unshaded; two hues that stay
apart for colour-blind readers.
"""

import pathlib

import numpy as np
from matplotlib import figure

from minifsd.eval import harness

# nyuv2-val positions (load order) of the three images.
PICKS = (119, 123, 1)
MODELS = (
    ("floor-segformer-b0", "SegFormer, v9"),
    ("floor-segformer-b0-v10", "SegFormer, v10"),
    ("floor-dinov2-s-v10", "DINOv2 frozen, v10"),
)
FOUND = np.array([0x2A, 0x78, 0xD6]) / 255  # reference palette blue
FALSE = np.array([0xEB, 0x68, 0x34]) / 255  # reference palette orange
ALPHA = 0.55
OUT = pathlib.Path("docs/report/figures/e2-overlays.pdf")


def overlay(rgb: np.ndarray, pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """The photo with found floor blue and false floor orange."""
    out = rgb.astype(np.float64) / 255
    for mask, colour in ((pred & truth, FOUND), (pred & ~truth, FALSE)):
        out[mask] = (1 - ALPHA) * out[mask] + ALPHA * colour
    return out


def main() -> None:
    """Writes OUT."""
    samples = list(harness.load_dataset("nyuv2-val"))
    picked = [samples[i] for i in PICKS]
    models = [
        harness.make_candidate(name, device="cpu", threads=2).predict
        for name, _ in MODELS
    ]
    fig = figure.Figure(figsize=(7.0, 4.0), dpi=200, layout="constrained")
    axes = fig.subplots(len(picked), 1 + len(models))
    titles = ["Photo", *(label for _, label in MODELS)]
    for row, sample in enumerate(picked):
        truth = sample.floor & sample.labeled
        panels = [sample.rgb / 255]
        panels += [overlay(sample.rgb, m(sample.rgb), truth) for m in models]
        for col, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
