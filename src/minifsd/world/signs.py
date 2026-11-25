"""Room numbers on the sign plates beside each door (D7).

MuJoCo has no text rendering, so each sign is a texture: the room number
(and sometimes a word such as "Lab" above it) drawn with OpenCV, then copied
into the compiled model's texture memory before the first render. The sign
reader (M5) reads these with off-the-shelf OCR.

The style (plate and text colours, font, word) belongs to the building, as
in a real one, so it is part of mjcf.Appearance; only the number changes
from sign to sign.
"""

from typing import TYPE_CHECKING

import cv2
import mujoco
import numpy as np

from minifsd.world import floor_plan

if TYPE_CHECKING:  # mjcf imports this module, so only for type hints
    from minifsd.world import mjcf

# The plate is 30 x 20 cm (mjcf.SIGN_HALF_SIZE): 2 mm per texel.
TEXTURE_WIDTH = 150
TEXTURE_HEIGHT = 100
MARGIN = 8  # texels of plain plate around the text

# OpenCV's Hershey fonts by name. The script fonts are left out: real room
# signs use plain lettering.
FONTS = {
    "simplex": cv2.FONT_HERSHEY_SIMPLEX,
    "plain": cv2.FONT_HERSHEY_PLAIN,
    "duplex": cv2.FONT_HERSHEY_DUPLEX,
    "complex": cv2.FONT_HERSHEY_COMPLEX,
    "triplex": cv2.FONT_HERSHEY_TRIPLEX,
    "complex_small": cv2.FONT_HERSHEY_COMPLEX_SMALL,
}
# Words some buildings put above the number. "" means no word.
WORDS = ("Room", "Lab", "Office", "Conf")
WORD_SHARE = 0.35  # share of the text area's height the word gets


def texture_name(number: int) -> str:
    """The MJCF texture name of a room's sign."""
    return f"sign_tex_{number}"


def material_name(number: int) -> str:
    """The MJCF material name of a room's sign."""
    return f"sign_mat_{number}"


def draw_sign(number: int, look: "mjcf.Appearance") -> np.ndarray:
    """Draws one sign plate.

    Args:
        number: The room number.
        look: The building's look; its sign_* fields set the style.

    Returns:
        (TEXTURE_HEIGHT, TEXTURE_WIDTH, 3) uint8 RGB, row 0 at the top.
    """
    plate = tuple(round(c * 255) for c in look.sign_rgb)
    text = tuple(round(c * 255) for c in look.sign_text_rgb)
    img = np.empty((TEXTURE_HEIGHT, TEXTURE_WIDTH, 3), np.uint8)
    img[:] = plate
    font = FONTS[look.sign_font]

    left, top = MARGIN, MARGIN
    width = TEXTURE_WIDTH - 2 * MARGIN
    height = TEXTURE_HEIGHT - 2 * MARGIN
    if look.sign_word:
        word_h = round(height * WORD_SHARE)
        gap = 4
        _put_fitted(img, look.sign_word, font, text, left, top, width, word_h)
        top += word_h + gap
        height -= word_h + gap
    _put_fitted(img, str(number), font, text, left, top, width, height)
    return img


def _put_fitted(
    img: np.ndarray,
    text: str,
    font: int,
    colour: tuple[int, ...],
    left: int,
    top: int,
    width: int,
    height: int,
) -> None:
    """Draws text as large as fits in the box, centred in it."""
    # The size OpenCV reports leaves out about half the stroke on each side,
    # and antialiasing adds a texel, so the fit allows for both.
    scale = 4.0
    while True:
        thickness = max(1, round(scale * 1.6))
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        pad = thickness + 1
        if (w + pad <= width and h + pad <= height) or scale < 0.2:
            break
        scale *= 0.95
    x = left + (width - w) // 2
    y = top + (height + h) // 2  # putText's origin is the baseline
    cv2.putText(img, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def paint(
    model: mujoco.MjModel, plan: floor_plan.FloorPlan, look: "mjcf.Appearance"
) -> None:
    """Copies every room's sign drawing into the compiled model's textures.

    Call it before the first render: a Renderer uploads textures to the GPU
    when it is created.

    Args:
        model: A model built by mjcf.build_mjcf for this plan.
        plan: The floor plan, for the room numbers.
        look: The look the model was built with.
    """
    for room in plan.rooms:
        tex = model.texture(texture_name(room.number))
        start = int(tex.adr[0])
        n = TEXTURE_WIDTH * TEXTURE_HEIGHT * 3
        model.tex_data[start : start + n] = draw_sign(room.number, look).ravel()
