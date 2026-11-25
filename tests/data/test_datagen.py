"""Tests for minifsd.data.datagen: sim frames with exact labels."""

import colorsys
import dataclasses
import json
import pathlib

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.data import camera_effects, datagen
from minifsd.geometry import frames
from minifsd.world import floor_plan, mjcf, sim

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=2)


def _distance_to_walls(plan: floor_plan.FloorPlan, x: float, y: float) -> float:
    def to_box(w: floor_plan.Wall) -> float:
        dx = max(w.x_min - x, 0.0, x - w.x_max)
        dy = max(w.y_min - y, 0.0, y - w.y_max)
        return float(np.hypot(dx, dy))

    return min(to_box(w) for w in plan.walls)


@settings(max_examples=20, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=5),
)
def test_poses_are_clear_of_walls_and_inside(seed: int, n: int) -> None:
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    poses = datagen.sample_poses(plan, 30, np.random.default_rng(seed))
    assert len(poses) == 30
    xs = [w.x_min for w in plan.walls] + [w.x_max for w in plan.walls]
    ys = [w.y_min for w in plan.walls] + [w.y_max for w in plan.walls]
    for p in poses:
        assert min(xs) < p.x < max(xs) and min(ys) < p.y < max(ys)
        # A cell of slack: the grid is 5 cm, the clearance is exact.
        assert _distance_to_walls(plan, p.x, p.y) >= datagen.CLEARANCE - 0.05
        assert -np.pi <= p.yaw < np.pi


def test_poses_are_reproducible() -> None:
    a = datagen.sample_poses(PLAN, 5, np.random.default_rng(3))
    b = datagen.sample_poses(PLAN, 5, np.random.default_rng(3))
    assert a == b


@given(st.integers(min_value=1, max_value=10_000))
def test_no_seed_is_in_two_splits(n: int) -> None:
    splits = {name: set(datagen.seeds_for(name, n)) for name in datagen.SPLITS}
    assert all(len(seeds) == n for seeds in splits.values())
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not splits[a] & splits[b], (a, b)


def test_rejects_an_unknown_split() -> None:
    with pytest.raises(ValueError):
        datagen.seeds_for("holdout", 3)


@pytest.mark.gl
def test_frames_have_aligned_labels() -> None:
    poses = datagen.sample_poses(PLAN, 3, np.random.default_rng(0))
    frames = list(datagen.render_frames(PLAN, poses))
    assert len(frames) == 3
    h, w = mjcf.IMAGE_HEIGHT, mjcf.IMAGE_WIDTH
    for f in frames:
        assert f.rgb.shape == (h, w, 3) and f.rgb.dtype == np.uint8
        assert f.floor.shape == (h, w) and f.floor.dtype == np.bool_
        assert f.depth.shape == (h, w) and f.depth.dtype == np.float32
        assert f.floor.any()  # a level camera always sees some floor
        assert f.plan_seed == PLAN.seed


def _read(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    assert image is not None, path
    return image


@pytest.mark.gl
def test_write_round_trips(tmp_path: pathlib.Path) -> None:
    poses = datagen.sample_poses(PLAN, 2, np.random.default_rng(0))
    frames = list(datagen.render_frames(PLAN, poses))
    for f in frames:
        datagen.write_frame(f, tmp_path)
    lines = (tmp_path / "meta.jsonl").read_text().splitlines()
    assert len(lines) == 2
    meta = json.loads(lines[1])
    f = frames[1]
    assert meta["plan_seed"] == PLAN.seed and meta["index"] == f.index
    assert meta["pose"] == pytest.approx([f.pose.x, f.pose.y, f.pose.yaw])
    stem = tmp_path / meta["stem"]
    rgb = cv2.cvtColor(_read(f"{stem}_rgb.png"), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(rgb, f.rgb)
    floor = _read(f"{stem}_floor.png") > 0
    np.testing.assert_array_equal(floor, f.floor)
    depth_mm = _read(f"{stem}_depth.png")
    assert depth_mm.dtype == np.uint16
    # Millimetres: within half a millimetre of the float depth.
    np.testing.assert_allclose(depth_mm / 1000, f.depth, atol=5e-4)


def test_poses_stay_clear_of_clutter() -> None:
    from minifsd.world import clutter

    plan = floor_plan.generate_floor_plan(seed=1, rooms_per_side=4)
    boxes = clutter.generate_clutter(plan, 8, np.random.default_rng(1))
    poses = datagen.sample_poses(plan, 200, np.random.default_rng(2), boxes)
    for p in poses:
        for b in boxes:
            assert not clutter.contains(b, p.x, p.y, datagen.CLEARANCE - 0.05)


@pytest.mark.gl
def test_clutter_is_not_floor() -> None:
    from minifsd.world import clutter

    # A box straight ahead of the camera: the centre pixel sees its near
    # face, 1.2 m away, and is not floor.
    box = clutter.Box(2.5, 0.0, 0.0, 0.3, 0.6, 1.0, (0.5, 0.4, 0.3, 1.0))
    pose = floor_plan.Pose2D(1.0, 0.0, 0.0)
    (frame,) = datagen.render_frames(PLAN, [pose], (box,))
    centre = frame.depth[mjcf.IMAGE_HEIGHT // 2, mjcf.IMAGE_WIDTH // 2]
    assert centre == pytest.approx(box.x - box.half_x - pose.x, abs=0.05)
    assert not frame.floor[mjcf.IMAGE_HEIGHT // 2, mjcf.IMAGE_WIDTH // 2]


@pytest.mark.gl
def test_read_frames_gives_back_what_was_written(
    tmp_path: pathlib.Path,
) -> None:
    poses = datagen.sample_poses(PLAN, 3, np.random.default_rng(0))
    frames = list(datagen.render_frames(PLAN, poses))
    for f in frames:
        datagen.write_frame(f, tmp_path)
    back = list(datagen.read_frames(tmp_path))
    assert [b.index for b in back] == [0, 1, 2]
    for f, b in zip(frames, back, strict=True):
        np.testing.assert_array_equal(b.rgb, f.rgb)
        np.testing.assert_array_equal(b.floor, f.floor)
        np.testing.assert_allclose(b.depth, f.depth, atol=5e-4)
        assert b.labeled.all()  # every sim pixel is labelled


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_sampled_cameras_stay_in_range(seed: int) -> None:
    cams = datagen.sample_cameras(50, np.random.default_rng(seed))
    assert len(cams) == 50
    low, high = datagen.CAMERA_HEIGHT_RANGE
    for c in cams:
        assert low <= c.height <= high
        assert 0.0 <= c.pitch <= datagen.MAX_PITCH
        assert abs(c.roll) <= datagen.MAX_ROLL
        lo, hi = datagen.FOVY_RANGE_DEG
        assert lo <= c.fovy_deg <= hi


def test_sampled_cameras_are_reproducible() -> None:
    a = datagen.sample_cameras(5, np.random.default_rng(3))
    b = datagen.sample_cameras(5, np.random.default_rng(3))
    assert a == b


@pytest.mark.gl
def test_meta_records_each_frames_camera(tmp_path: pathlib.Path) -> None:
    poses = datagen.sample_poses(PLAN, 2, np.random.default_rng(0))
    cams = [sim.CameraSetup(1.4, 0.2, -0.05), sim.CameraSetup(0.4, 0.0, 0.03)]
    frames = list(datagen.render_frames(PLAN, poses, cameras=cams))
    assert [f.camera for f in frames] == cams
    for f in frames:
        datagen.write_frame(f, tmp_path)
    lines = (tmp_path / "meta.jsonl").read_text().splitlines()
    for line, cam in zip(lines, cams, strict=True):
        meta = json.loads(line)
        assert meta["camera_height"] == pytest.approx(cam.height)
        assert meta["camera_pitch"] == pytest.approx(cam.pitch)
        assert meta["camera_roll"] == pytest.approx(cam.roll)


@pytest.mark.gl
def test_meta_records_each_frames_own_intrinsics(
    tmp_path: pathlib.Path,
) -> None:
    poses = datagen.sample_poses(PLAN, 2, np.random.default_rng(0))
    cams = [
        sim.CameraSetup(0.5, 0.0, 0.0, 55.0),
        sim.CameraSetup(0.5, 0.0, 0.0, 72.0),
    ]
    for f in datagen.render_frames(PLAN, poses, cameras=cams):
        datagen.write_frame(f, tmp_path)
    lines = (tmp_path / "meta.jsonl").read_text().splitlines()
    for line, cam in zip(lines, cams, strict=True):
        k = frames.intrinsics_from_fovy(
            mjcf.IMAGE_WIDTH, mjcf.IMAGE_HEIGHT, cam.fovy_deg
        )
        assert json.loads(line)["intrinsics"] == pytest.approx(
            dataclasses.asdict(k)
        )


def _muted(rgb: tuple[float, float, float]) -> bool:
    return colorsys.rgb_to_hsv(*rgb)[1] <= datagen.MUTED_MAX_SATURATION


def _colours(a: mjcf.Appearance) -> list[tuple[float, float, float]]:
    return [a.floor_rgb, a.floor_rgb2, a.wall_rgb, a.ceiling_rgb]


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_sampled_looks_stay_in_range(seed: int) -> None:
    # Appearance itself rejects colours outside [0, 1] and unknown kinds,
    # so building one is half the check.
    looks = datagen.sample_appearances(50, np.random.default_rng(seed))
    assert len(looks) == 50
    low, high = datagen.FLOOR_PERIOD_RANGE_M
    for a in looks:
        assert a.floor_kind in mjcf.FLOOR_KINDS
        assert low <= a.floor_period_m <= high
        # The second floor colour must differ from the first, or a checker
        # or carpet floor quietly turns plain.
        if a.floor_kind != "plain":
            assert a.floor_rgb != a.floor_rgb2


def test_sampled_looks_are_mostly_muted_with_a_wild_share() -> None:
    # Option (c): most buildings in muted colours, about WILD_FRACTION in
    # any colour at all. A wild look can come out muted by chance, so the
    # muted share is at least 1 - WILD_FRACTION, but not much more.
    looks = datagen.sample_appearances(2000, np.random.default_rng(0))
    muted = np.mean([all(_muted(c) for c in _colours(a)) for a in looks])
    expected = 1 - datagen.WILD_FRACTION
    assert expected - 0.04 <= muted <= expected + 0.06
    # And every floor kind turns up.
    assert {a.floor_kind for a in looks} == set(mjcf.FLOOR_KINDS)


def test_sampled_looks_are_reproducible() -> None:
    a = datagen.sample_appearances(5, np.random.default_rng(3))
    b = datagen.sample_appearances(5, np.random.default_rng(3))
    c = datagen.sample_appearances(5, np.random.default_rng(4))
    assert a == b
    assert a != c


@pytest.mark.gl
def test_meta_records_the_plans_look(tmp_path: pathlib.Path) -> None:
    poses = datagen.sample_poses(PLAN, 2, np.random.default_rng(0))
    look = mjcf.Appearance(floor_kind="carpet", floor_rgb2=(0.3, 0.3, 0.3))
    frames = list(datagen.render_frames(PLAN, poses, appearance=look))
    assert all(f.appearance == look for f in frames)
    for f in frames:
        datagen.write_frame(f, tmp_path)
    for line in (tmp_path / "meta.jsonl").read_text().splitlines():
        meta = json.loads(line)["appearance"]
        assert meta["floor_kind"] == "carpet"
        assert meta["floor_rgb2"] == pytest.approx([0.3, 0.3, 0.3])


def test_about_half_the_sampled_plans_have_glass() -> None:
    looks = datagen.sample_appearances(2000, np.random.default_rng(0))
    shares = np.array([a.glass_share for a in looks])
    assert ((shares >= 0.0) & (shares <= 1.0)).all()
    with_glass = (shares > 0).mean()
    assert abs(with_glass - datagen.GLASS_PLAN_SHARE) < 0.04
    low, high = datagen.GLASS_SHARE_RANGE
    assert ((shares[shares > 0] >= low) & (shares[shares > 0] <= high)).all()


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_sampled_lighting_stays_in_range(seed: int) -> None:
    for a in datagen.sample_appearances(50, np.random.default_rng(seed)):
        assert datagen.AMBIENT_RANGE[0] <= a.ambient <= datagen.AMBIENT_RANGE[1]
        lo, hi = datagen.MAIN_LIGHT_RANGE
        assert lo <= a.main_light <= hi
        lo, hi = datagen.SIDE_LIGHT_RANGE
        assert lo <= a.side_light <= hi
        x, y, z = a.side_light_dir
        assert np.linalg.norm([x, y, z]) == pytest.approx(1.0)
        # Elevation below the horizon, within the sampled range.
        down = np.degrees(np.arcsin(-z))
        lo, hi = datagen.SIDE_LIGHT_ELEVATION_DEG
        assert lo - 1e-6 <= down <= hi + 1e-6


@pytest.mark.gl
def test_sampled_lighting_is_never_black_or_blown_out() -> None:
    # D4's real test: whatever the draw, a model gets a usable picture. The
    # darkest floor (value 0.15) under the dimmest lights must still show,
    # and the palest walls under the brightest must not clip to white.
    looks = datagen.sample_appearances(30, np.random.default_rng(0))
    pose = floor_plan.Pose2D(PLAN.spawn.x, PLAN.spawn.y, 0.3)
    cam = sim.CameraSetup(1.0, np.radians(10.0), 0.0)
    for a in looks:
        s = sim.Sim(PLAN, appearance=a)
        try:
            s.set_pose(pose)
            s.set_camera(cam)
            rgb = s.render_rgb()
        finally:
            s.close()
        grey = rgb.mean(axis=2)
        assert 30 <= grey.mean() <= 225, a
        assert (grey < 10).mean() < 0.2, a  # crushed to black
        assert (rgb.min(axis=2) >= 250).mean() < 0.05, a  # clipped to white


def _plain_frame() -> datagen.Frame:
    rng = np.random.default_rng(0)
    return datagen.Frame(
        plan_seed=7,
        index=0,
        pose=floor_plan.Pose2D(1.0, 0.0, 0.0),
        rgb=rng.integers(0, 256, (24, 32, 3), np.uint8),
        depth=rng.uniform(0.5, 5.0, (24, 32)).astype(np.float32),
        floor=rng.random((24, 32)) > 0.5,
    )


def test_camera_effects_change_the_picture_and_not_the_labels() -> None:
    frame = _plain_frame()
    effects = camera_effects.Effects(blur_sigma=1.0, noise_sigma=5.0)
    out = datagen.with_effects(frame, effects, np.random.default_rng(0))
    assert out.effects == effects
    assert not np.array_equal(out.rgb, frame.rgb)
    np.testing.assert_array_equal(out.depth, frame.depth)
    np.testing.assert_array_equal(out.floor, frame.floor)
    assert (out.plan_seed, out.index, out.pose) == (7, 0, frame.pose)


def test_meta_records_the_frames_camera_effects(tmp_path: pathlib.Path) -> None:
    effects = camera_effects.Effects(motion_px=3.0, jpeg_quality=60)
    frame = datagen.with_effects(
        _plain_frame(), effects, np.random.default_rng(0)
    )
    datagen.write_frame(frame, tmp_path)
    meta = json.loads((tmp_path / "meta.jsonl").read_text())["effects"]
    assert meta["motion_px"] == pytest.approx(3.0)
    assert meta["jpeg_quality"] == 60
