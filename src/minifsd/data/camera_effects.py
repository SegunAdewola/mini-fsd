"""What a real camera does to a picture, applied to sim renders (D5).

MuJoCo renders are perfectly sharp, clean and exposed. Phone photos are not:
the lens defocuses, a moving hand smears, the sensor adds noise, auto
exposure lands too bright or too dark, and the photo is saved as JPEG. A
model trained only on clean renders can lean on that cleanness, so the data
engine applies these effects to each frame, in the order a camera does:

    optics (defocus, motion blur) -> sensor (exposure gain, noise)
    -> image processing (gamma) -> JPEG

Only the RGB image changes. Labels (floor mask, depth) describe the scene,
not the picture, so they never pass through here.
"""

import dataclasses
import math

import cv2
import numpy as np

# Sampling ranges. Each effect is on in only part of the frames, so models
# still see sharp, clean images as well.
BLUR_CHANCE = 0.5
BLUR_SIGMA_MAX = 1.5  # pixels, Gaussian defocus at 320 x 240
MOTION_CHANCE = 0.3
MOTION_PX_MAX = 6.0  # smear length in pixels
NOISE_SIGMA_MAX = 8.0  # on the 0 to 255 scale
EXPOSURE_RANGE = (0.6, 1.5)  # gain on linear brightness
GAMMA_RANGE = (0.8, 1.25)  # out = in ** gamma, on [0, 1]
JPEG_CHANCE = 0.7
JPEG_QUALITY_RANGE = (40, 95)


@dataclasses.dataclass(frozen=True)
class Effects:
    """One frame's camera effects. The default leaves the image as it is.

    Attributes:
        blur_sigma: Gaussian defocus in pixels; 0 is off.
        motion_px: Length of a straight-line motion smear in pixels; 0 is
            off.
        motion_angle: Direction of the smear in radians, 0 along the image
            rows.
        noise_sigma: Standard deviation of Gaussian sensor noise, on the
            0 to 255 scale; 0 is off.
        exposure: Brightness gain; 1 is off.
        gamma: out = in ** gamma on [0, 1]; below 1 brightens the mid-tones,
            above 1 darkens them; 1 is off.
        jpeg_quality: JPEG quality 1 to 100, or None for no compression.
    """

    blur_sigma: float = 0.0
    motion_px: float = 0.0
    motion_angle: float = 0.0
    noise_sigma: float = 0.0
    exposure: float = 1.0
    gamma: float = 1.0
    jpeg_quality: int | None = None

    def __post_init__(self) -> None:
        """Rejects settings that make no sense.

        Raises:
            ValueError: For a negative blur, smear or noise, a gain or gamma
                that isn't positive, or a JPEG quality outside 1 to 100.
        """
        for name in ("blur_sigma", "motion_px", "noise_sigma"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be >= 0, got {getattr(self, name)}"
                )
        for name in ("exposure", "gamma"):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} must be > 0, got {getattr(self, name)}"
                )
        q = self.jpeg_quality
        if q is not None and not 1 <= q <= 100:
            raise ValueError(f"jpeg_quality must be 1 to 100, got {q}")


def sample_effects(rng: np.random.Generator) -> Effects:
    """Random effects for one frame, each on with its own chance.

    Args:
        rng: The random source; the same state gives the same effects.

    Returns:
        The effects, within the module's ranges.
    """
    # Every value is drawn whether or not its effect is on, so turning one
    # effect off never shifts the draws of the others.
    blur_on = rng.random() < BLUR_CHANCE
    blur = rng.uniform(0.3, BLUR_SIGMA_MAX)
    motion_on = rng.random() < MOTION_CHANCE
    motion = rng.uniform(1.5, MOTION_PX_MAX)
    angle = rng.uniform(0.0, math.pi)
    noise = rng.uniform(0.0, NOISE_SIGMA_MAX)
    exposure = rng.uniform(*EXPOSURE_RANGE)
    gamma = rng.uniform(*GAMMA_RANGE)
    jpeg_on = rng.random() < JPEG_CHANCE
    quality = int(
        rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1)
    )
    return Effects(
        blur_sigma=float(blur) if blur_on else 0.0,
        motion_px=float(motion) if motion_on else 0.0,
        motion_angle=float(angle) if motion_on else 0.0,
        noise_sigma=float(noise),
        exposure=float(exposure),
        gamma=float(gamma),
        jpeg_quality=quality if jpeg_on else None,
    )


def apply(
    rgb: np.ndarray, effects: Effects, rng: np.random.Generator
) -> np.ndarray:
    """Returns a copy of the image as a camera with these effects saw it.

    Args:
        rgb: (H, W, 3) uint8.
        effects: What to apply.
        rng: The random source for the noise; the same state gives the
            same output.

    Returns:
        (H, W, 3) uint8, a new array; rgb is not changed.
    """
    if effects == Effects():
        return rgb.copy()
    img: np.ndarray = rgb.astype(np.float32) / 255.0
    if effects.blur_sigma > 0:
        img = cv2.GaussianBlur(img, (0, 0), effects.blur_sigma)
    if effects.motion_px > 0:
        img = cv2.filter2D(
            img, -1, _motion_kernel(effects.motion_px, effects.motion_angle)
        )
    img = img * effects.exposure
    if effects.noise_sigma > 0:
        img = img + rng.normal(0.0, effects.noise_sigma / 255.0, img.shape)
    img = np.clip(img, 0.0, 1.0) ** effects.gamma
    out = np.round(img * 255.0).astype(np.uint8)
    if effects.jpeg_quality is not None:
        out = _jpeg(out, effects.jpeg_quality)
    return out


def _motion_kernel(length: float, angle: float) -> np.ndarray:
    """A normalized line kernel: a smear of this length and direction."""
    size = int(math.ceil(length)) | 1  # odd, so the line is centred
    kernel = np.zeros((size, size), np.float32)
    c = size // 2
    dx = 0.5 * (length - 1) * math.cos(angle)
    dy = 0.5 * (length - 1) * math.sin(angle)
    # Sample points along the line rather than cv2.line, so a fractional
    # length still gives the right total smear.
    for t in np.linspace(-1.0, 1.0, 4 * size):
        x, y = round(c + t * dx), round(c + t * dy)
        kernel[y, x] = 1.0
    return kernel / kernel.sum()


def _jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    """Encodes and decodes as JPEG, keeping the compression artefacts."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("OpenCV could not encode the image as JPEG")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV could not decode its own JPEG")
    return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB))
