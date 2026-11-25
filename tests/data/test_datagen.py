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

from minifsd.data import boxes, camera_effects, datagen
from minifsd.geometry import frames
from minifsd.world import clutter, floor_plan, mjcf, signs, sim

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


@settings(max_examples=20, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=5),
)
def test_poses_look_along_open_space(seed: int, n: int) -> None:
    # D6: in trial batches 7 to 10% of frames showed under 2% floor: a
    # camera facing a wall, often a tall one held level, whose view meets
    # the floor only 2 to 3 m ahead. So each pose needs free space along
    # its heading up to where its camera's lowest ray reaches the floor,
    # plus MIN_CLEAR_AHEAD of floor beyond that.
    plan = floor_plan.generate_floor_plan(seed, rooms_per_side=n)
    rng = np.random.default_rng(seed)
    boxes = clutter.generate_clutter(plan, 4, rng)
    cams = datagen.sample_cameras(30, rng)
    poses = datagen.sample_poses(plan, 30, rng, boxes, cams)
    slack = 0.05  # one occupancy cell
    for p, cam in zip(poses, cams, strict=True):
        lowest_ray = cam.pitch + np.radians(cam.fovy_deg) / 2
        near_edge = cam.height / np.tan(lowest_ray)
        reach = near_edge + datagen.MIN_CLEAR_AHEAD
        for t in np.arange(0.0, reach - slack, 0.02):
            x, y = p.x + t * np.cos(p.yaw), p.y + t * np.sin(p.yaw)
            assert _distance_to_walls(plan, x, y) > 0, (p, cam, t)
            assert not any(clutter.contains(b, x, y) for b in boxes), (p, t)


def _luma(rgb: tuple[float, float, float]) -> float:
    r, g, b = rgb
    return 0.299 * r + 0.587 * g + 0.114 * b


def test_sampled_signs_are_readable_and_varied() -> None:
    # D7: one sign style per building. Text must stand out from its plate
    # whatever the draw, or the sign reader has nothing to read.
    looks = datagen.sample_appearances(1000, np.random.default_rng(0))
    for a in looks:
        contrast = abs(_luma(a.sign_rgb) - _luma(a.sign_text_rgb))
        assert contrast >= datagen.MIN_SIGN_CONTRAST, a
    assert {a.sign_font for a in looks} == set(signs.FONTS)
    worded = np.mean([a.sign_word != "" for a in looks])
    assert abs(worded - datagen.SIGN_WORD_CHANCE) < 0.05
    dark_text = np.mean(
        [_luma(a.sign_text_rgb) < _luma(a.sign_rgb) for a in looks]
    )
    assert 0.4 < dark_text < 0.6  # both dark on light and light on dark


def _project(s: sim.Sim, points: np.ndarray) -> np.ndarray:
    """World points to pixel (col, row), written from MuJoCo's camera frame.

    A MuJoCo camera looks along its own -z with +y up in the image; cam_xmat
    holds its axes in world coordinates (columns).
    """
    cam = s.model.camera(mjcf.CAMERA).id
    k = s.intrinsics()
    centre = s.data.cam_xpos[cam]
    axes = s.data.cam_xmat[cam].reshape(3, 3)
    local = (points - centre) @ axes  # into the camera's own frame
    depth = -local[:, 2]
    cols = k.cx + k.fx * local[:, 0] / depth
    rows = k.cy - k.fy * local[:, 1] / depth
    return np.stack([cols, rows], axis=1)


def _corners(s: sim.Sim, geom: str) -> np.ndarray:
    g = s.data.geom(geom)
    size = s.model.geom(geom).size
    signs_ = np.array(
        [[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
    )
    return np.asarray(g.xpos + (signs_ * size) @ g.xmat.reshape(3, 3).T)


def _box_of(frame: datagen.Frame, instance: str) -> boxes.Box2D | None:
    return next((b for b in frame.boxes if b.instance == instance), None)


def _facing(
    room: floor_plan.Room, x: float, distance: float
) -> floor_plan.Pose2D:
    return floor_plan.Pose2D(x, -room.side * distance, room.side * np.pi / 2)


@pytest.mark.gl
def test_a_signs_box_matches_its_projected_corners() -> None:
    room = PLAN.rooms[0]
    x = room.door_x + PLAN.door_width / 2 + mjcf.SIGN_GAP
    pose = _facing(room, x - 0.4, 0.5)  # a little to one side, so not square on
    cam = sim.CameraSetup(1.3, 0.0, 0.0)
    (frame,) = datagen.render_frames(PLAN, [pose], cameras=[cam])
    b = _box_of(frame, f"sign_{room.number}")
    assert b is not None and b.category == "room_sign"
    s = sim.Sim(PLAN)
    try:
        s.set_pose(pose)
        s.set_camera(cam)
        uv = _project(s, _corners(s, f"sign_{room.number}"))
    finally:
        s.close()
    # Pixel edges against projected corners: within 2 px allows for the
    # rasterizer's half-pixel rule and the sign's thickness.
    assert b.x == pytest.approx(uv[:, 0].min(), abs=2)
    assert b.x + b.w == pytest.approx(uv[:, 0].max(), abs=2)
    assert b.y == pytest.approx(uv[:, 1].min(), abs=2)
    assert b.y + b.h == pytest.approx(uv[:, 1].max(), abs=2)


@pytest.mark.gl
def test_a_doors_box_spans_its_frame_jamb_to_jamb() -> None:
    room = PLAN.rooms[0]
    pose = _facing(room, room.door_x, 1.0)
    cam = sim.CameraSetup(1.0, np.radians(5.0), 0.0, 70.0)
    (frame,) = datagen.render_frames(PLAN, [pose], cameras=[cam])
    b = _box_of(frame, f"door_{room.number}")
    assert b is not None and b.category == "door"
    s = sim.Sim(PLAN)
    try:
        s.set_pose(pose)
        s.set_camera(cam)
        left = _project(s, _corners(s, f"door_{room.number}_jamb_l"))
        right = _project(s, _corners(s, f"door_{room.number}_jamb_r"))
    finally:
        s.close()
    cols = np.concatenate([left[:, 0], right[:, 0]])
    assert b.x == pytest.approx(max(cols.min(), 0), abs=2)
    assert b.x + b.w == pytest.approx(min(cols.max(), mjcf.IMAGE_WIDTH), abs=2)
    # Floor to head, or the image's bottom edge if the floor is out of view.
    assert b.y + b.h >= mjcf.IMAGE_HEIGHT - 2 or b.h > 0.5 * mjcf.IMAGE_HEIGHT


@pytest.mark.gl
def test_an_obstacle_behind_glass_gets_a_box_and_a_hidden_one_does_not() -> (
    None
):
    room = PLAN.rooms[0]
    # One box inside the room behind the glass front, one behind a solid
    # wall: the back wall of the room across the hallway.
    inside = clutter.Box(
        room.door_x + 1.0,
        room.side * 2.0,
        0.0,
        0.3,
        0.3,
        0.8,
        (0.8, 0.2, 0.2, 1),
    )
    other = next(r for r in PLAN.rooms if r.side != room.side)
    hidden = clutter.Box(
        other.door_x, other.side * 2.0, 0.0, 0.3, 0.3, 0.8, (0.2, 0.2, 0.8, 1)
    )
    glass = mjcf.Appearance(glass_share=1.0)
    pose = _facing(room, room.door_x + 1.0, 0.4)
    (frame,) = datagen.render_frames(
        PLAN,
        [pose],
        clutter=(inside, hidden),
        appearance=glass,
        cameras=[sim.CameraSetup(1.0, np.radians(10.0), 0.0, 70.0)],
    )
    assert _box_of(frame, "clutter_0") is not None
    assert _box_of(frame, "clutter_1") is None


@pytest.mark.gl
def test_boxes_go_into_meta_and_the_splits_coco_file(
    tmp_path: pathlib.Path,
) -> None:
    room = PLAN.rooms[0]
    poses = [_facing(room, room.door_x, 1.0), PLAN.spawn]
    frames = list(datagen.render_frames(PLAN, poses))
    assert frames[0].boxes  # facing a door, it sees the door
    for f in frames:
        datagen.write_frame(f, tmp_path)
    datagen.write_coco(tmp_path)
    coco = json.loads((tmp_path / "coco.json").read_text())
    assert [c["name"] for c in coco["categories"]] == list(boxes.CATEGORIES)
    assert len(coco["images"]) == 2
    assert coco["images"][0]["file_name"].endswith("_rgb.png")
    assert len(coco["annotations"]) == sum(len(f.boxes) for f in frames)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        assert x >= 0 and x + w <= mjcf.IMAGE_WIDTH
        assert y >= 0 and y + h <= mjcf.IMAGE_HEIGHT


def test_door_frames_vary_from_building_to_building() -> None:
    looks = datagen.sample_appearances(200, np.random.default_rng(0))
    frames_ = {a.door_frame_rgb for a in looks}
    assert len(frames_) == 200  # a colour per building
    values = [max(c) for c in frames_]
    lo, hi = datagen.DOOR_FRAME_VALUE_RANGE
    assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in values)
