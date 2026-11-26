"""Losses for HydraNet-lite: one per head, each with its own mask (H2, E4).

Training mixes v10 frames, which have every label, with COCO photos, which
have boxes only, and only for some classes. So every loss takes a mask of
what each image actually labels, and an image adds nothing for a task or a
class it doesn't label:

- Heatmaps: CenterNet's focal loss (Zhou et al., 2019), normalised by the
  number of object centres. A per-image, per-class mask lets COCO images
  leave door and room sign out: their doors are unlabeled, not absent, so
  they must not be learned as background (the problem E1's detectors have).
- Box size and centre offset: L1 at object centres only.
- Floor: cross-entropy on logits scaled up to the label's size, as in E2.
- Depth: the scale-invariant log loss of Eigen et al. (2014) with
  lambda = 1, the variance of the log error. A global scale costs nothing:
  the harness fits scale, and a phone's camera isn't the sim's.
"""

import torch
from torch.nn import functional

EPS = 1e-4  # keeps the focal loss's logs finite


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, class_mask: torch.Tensor
) -> torch.Tensor:
    """CenterNet's focal loss on heatmaps.

    Args:
        logits: (B, C, H, W) heatmap logits.
        target: (B, C, H, W) Gaussian targets, 1 at object centres.
        class_mask: (B, C) bool, True where the image labels that class.
    """
    p = torch.sigmoid(logits).clamp(EPS, 1 - EPS)
    centre = target.eq(1.0)
    pos = torch.log(p) * (1 - p) ** 2 * centre
    neg = torch.log(1 - p) * p**2 * (1 - target) ** 4 * ~centre
    keep = class_mask[:, :, None, None].to(logits.dtype)
    n_centres = (centre * keep).sum().clamp(min=1.0)
    return -((pos + neg) * keep).sum() / n_centres


def masked_l1(
    pred: torch.Tensor, target: torch.Tensor, centres: torch.Tensor
) -> torch.Tensor:
    """Mean absolute error over the (B, H, W) centre cells only."""
    keep = centres[:, None].expand_as(pred)
    if not keep.any():
        return pred.sum() * 0.0  # zero, but still part of the graph
    return (pred - target).abs()[keep].mean()


def floor_loss(
    logits: torch.Tensor, labels: torch.Tensor, has_floor: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy on the images that label floor.

    Args:
        logits: (B, 2, h, w) floor logits.
        labels: (B, H, W) 0 or 1.
        has_floor: (B,) bool.
    """
    if not has_floor.any():
        return logits.sum() * 0.0
    big = functional.interpolate(
        logits[has_floor],
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return functional.cross_entropy(big, labels[has_floor])


def depth_loss(
    log_pred: torch.Tensor, depth: torch.Tensor, has_depth: torch.Tensor
) -> torch.Tensor:
    """Scale-invariant log loss, averaged over images that label depth.

    Args:
        log_pred: (B, 1, h, w) predicted log depth.
        depth: (B, H, W) metres; 0 where there is no reading.
        has_depth: (B,) bool.
    """
    if log_pred.shape[-2:] != depth.shape[-2:]:
        log_pred = functional.interpolate(
            log_pred,
            size=depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    losses = []
    for i in torch.nonzero(has_depth).flatten().tolist():
        valid = depth[i] > 0
        if not valid.any():
            continue
        d = log_pred[i, 0][valid] - torch.log(depth[i][valid])
        losses.append((d**2).mean() - d.mean() ** 2)
    if not losses:
        return log_pred.sum() * 0.0
    return torch.stack(losses).mean()
