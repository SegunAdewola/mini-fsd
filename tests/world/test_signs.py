"""Tests for minifsd.world.signs: room numbers drawn on the sign plates (D7)."""

import dataclasses
import math

import mujoco
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.world import floor_plan, mjcf, signs, sim

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=2)
DEFAULT = mjcf.DEFAULT_APPEARANCE


def _ink(img: np.ndarray, look: mjcf.Appearance) -> np.ndarray:
    """Pixels nearer the text colour than the plate colour."""
    text = np.array(look.sign_text_rgb) * 255
    plate = np.array(look.sign_rgb) * 255
    d_text = np.linalg.norm(img - text, axis=2)
    d_plate = np.linalg.norm(img - plate, axis=2)
    return np.asarray(d_text < d_plate)


def test_a_sign_is_a_plate_with_text_on_it() -> None:
    img = signs.draw_sign(101, DEFAULT)
    assert img.shape == (signs.TEXTURE_HEIGHT, signs.TEXTURE_WIDTH, 3)
    assert img.dtype == np.uint8
    ink = _ink(img, DEFAULT)
    assert 0.03 < ink.mean() < 0.5  # some text, mostly plate


@settings(max_examples=40, deadline=None)
@given(
    st.integers(min_value=1, max_value=9949),
    st.sampled_from(tuple(signs.FONTS)),
    st.sampled_from(("", *signs.WORDS)),
)
def test_text_always_fits_inside_the_margin(
    number: int, font: str, word: str
) -> None:
    look = dataclasses.replace(DEFAULT, sign_font=font, sign_word=word)
    ink = _ink(signs.draw_sign(number, look), look)
    rows, cols = np.nonzero(ink)
    assert len(rows) > 0
    m = signs.MARGIN
    h, w = ink.shape
    assert rows.min() >= m and rows.max() < h - m
    assert cols.min() >= m and cols.max() < w - m


def test_each_number_gets_its_own_sign() -> None:
    a, b = signs.draw_sign(101, DEFAULT), signs.draw_sign(102, DEFAULT)
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(a, signs.draw_sign(101, DEFAULT))


def test_a_word_goes_above_the_number() -> None:
    look = dataclasses.replace(DEFAULT, sign_word="Lab")
    plain = _ink(signs.draw_sign(101, DEFAULT), DEFAULT)
    worded = _ink(signs.draw_sign(101, look), look)
    top = slice(0, signs.TEXTURE_HEIGHT // 3)
    assert worded[top].sum() > plain[top].sum()


def test_rejects_an_unknown_font_or_word() -> None:
    with pytest.raises(ValueError):
        mjcf.Appearance(sign_font="comic-sans")
    with pytest.raises(ValueError):
        mjcf.Appearance(sign_word="Toilets")


def test_the_model_holds_each_signs_own_texture() -> None:
    look = dataclasses.replace(DEFAULT, sign_word="Room", sign_font="duplex")
    s = sim.Sim(PLAN, appearance=look)
    try:
        for room in PLAN.rooms:
            tex = s.model.texture(signs.texture_name(room.number))
            n = int(tex.width[0] * tex.height[0] * tex.nchannel[0])
            data = s.model.tex_data[int(tex.adr[0]) : int(tex.adr[0]) + n]
            np.testing.assert_array_equal(
                data.reshape(signs.TEXTURE_HEIGHT, signs.TEXTURE_WIDTH, 3),
                signs.draw_sign(room.number, look),
            )
    finally:
        s.close()


def test_signs_stay_visual_only() -> None:
    model = mujoco.MjModel.from_xml_string(mjcf.build_mjcf(PLAN))
    for room in PLAN.rooms:
        g = model.geom(f"sign_{room.number}")
        assert g.contype[0] == 0 and g.conaffinity[0] == 0


def _sign_crop(s: sim.Sim, room: floor_plan.Room) -> np.ndarray:
    """Faces the sign from the hallway and returns its rendered pixels."""
    x = room.door_x + PLAN.door_width / 2 + mjcf.SIGN_GAP
    yaw = room.side * math.pi / 2  # +y for a left sign, -y for a right one
    s.set_pose(floor_plan.Pose2D(x, -room.side * 0.6, yaw))
    s.set_camera(sim.CameraSetup(height=mjcf.SIGN_HEIGHT))
    seg = s.render_segmentation()
    mask = seg == s.model.geom(f"sign_{room.number}").id
    rows, cols = np.nonzero(mask)
    assert len(rows) > 200  # the sign fills a good part of the view
    rgb = s.render_rgb()
    return rgb[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


@pytest.mark.gl
@pytest.mark.parametrize("side", [floor_plan.LEFT, floor_plan.RIGHT])
def test_signs_read_the_right_way_round_on_both_sides(side: int) -> None:
    # MuJoCo maps a texture onto a box face in one fixed direction, so a
    # sign on the other wall comes out mirrored unless it is turned round.
    # The rendered sign must look more like the drawing than like its
    # mirror image.
    import cv2

    room = next(r for r in PLAN.rooms if r.side == side)
    s = sim.Sim(PLAN)
    try:
        crop = _sign_crop(s, room)
    finally:
        s.close()
    # Grey, so the plate's colour (the same mirrored or not) can't decide it;
    # only where the text strokes are.
    drawn = cv2.cvtColor(
        signs.draw_sign(room.number, DEFAULT), cv2.COLOR_RGB2GRAY
    )
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    crop = cv2.resize(crop, drawn.shape[::-1])
    drawn, crop = drawn.astype(np.float32), crop.astype(np.float32)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))

    # Measured: about 0.91 as drawn against 0.74 to 0.80 mirrored. With the
    # half turn removed, a right-hand sign scores the other way round (0.74
    # against 0.91), so 0.05 separates the two cases with room to spare.
    assert corr(crop, drawn) > corr(crop, drawn[:, ::-1]) + 0.05
