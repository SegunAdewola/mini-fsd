"""Tests for minifsd.world.occupancy: floor plan to a map_server map.

map_server conventions (nav2_map_server): the PGM's first row is the top of
the map (largest y); black (0) is occupied and white (254) is free; the YAML
origin is the world position of the bottom-left pixel's corner.
"""

import json
import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from minifsd.world import floor_plan, occupancy

PLAN = floor_plan.generate_floor_plan(seed=0, rooms_per_side=2)
RES = occupancy.RESOLUTION


def _read_yaml(text: str) -> dict[str, object]:
    # The map YAML is flat "key: value" lines; values are numbers, a JSON
    # list (origin) or a bare string.
    meta: dict[str, object] = {}
    for line in text.splitlines():
        key, value = line.split(": ", 1)
        try:
            meta[key] = json.loads(value)
        except json.JSONDecodeError:
            meta[key] = value
    return meta


def _occupied(grid: occupancy.Grid, x: float, y: float) -> bool:
    row, col = grid.world_to_cell(x, y)
    return bool(grid.occupied[row, col])


def _wall_samples(wall: floor_plan.Wall) -> list[tuple[float, float]]:
    # Points well inside the wall, away from its edges.
    xs = np.linspace(wall.x_min, wall.x_max, 7)[1:-1]
    ys = np.linspace(wall.y_min, wall.y_max, 5)[1:-1]
    return [(float(x), float(y)) for x in xs for y in ys]


def _door_samples(
    plan: floor_plan.FloorPlan, room: floor_plan.Room
) -> list[tuple[float, float]]:
    # Points across the door opening, one cell in from each jamb, through
    # the whole thickness of the hallway wall.
    half = plan.hallway_width / 2
    reach = plan.door_width / 2 - 2 * RES
    xs = np.linspace(room.door_x - reach, room.door_x + reach, 5)
    ys = [half, half + floor_plan.WALL_THICKNESS / 2, half + 0.1]
    return [(float(x), room.side * y) for x in xs for y in ys]


def _check_plan(plan: floor_plan.FloorPlan) -> None:
    grid = occupancy.from_floor_plan(plan)
    for wall in plan.walls:
        for x, y in _wall_samples(wall):
            assert _occupied(grid, x, y), (wall, x, y)
    for room in plan.rooms:
        for x, y in _door_samples(plan, room):
            assert not _occupied(grid, x, y), (room.number, x, y)
        assert not _occupied(grid, room.goal.x, room.goal.y)
    assert not _occupied(grid, plan.spawn.x, plan.spawn.y)


def test_walls_occupied_doors_and_spawn_free() -> None:
    _check_plan(PLAN)


@settings(max_examples=30, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**32 - 1),
    st.integers(min_value=1, max_value=6),
)
def test_any_floor_plan(seed: int, n: int) -> None:
    _check_plan(floor_plan.generate_floor_plan(seed, rooms_per_side=n))


def test_grid_covers_every_wall_with_a_margin() -> None:
    grid = occupancy.from_floor_plan(PLAN)
    rows, cols = grid.occupied.shape
    x_max = grid.origin_x + cols * RES
    y_max = grid.origin_y + rows * RES
    for wall in PLAN.walls:
        assert grid.origin_x < wall.x_min and wall.x_max < x_max
        assert grid.origin_y < wall.y_min and wall.y_max < y_max
    # The border cells are outside the building, so free.
    assert not grid.occupied[0, :].any() and not grid.occupied[-1, :].any()


def test_cells_that_only_touch_a_wall_stay_free() -> None:
    # On a 5 cm grid, a 1 m by 0.1 m wall covers exactly 20 x 2 cells. If
    # cells that only touch its edges counted, every wall would grow by a
    # cell on each side and doors would shrink by 10 cm.
    grid = occupancy.rasterize([floor_plan.Wall(0.0, 0.0, 1.0, 0.1)])
    assert grid.occupied.sum() == 40


@given(st.floats(-2, 15), st.floats(-6, 6))
def test_cell_centre_is_within_half_a_cell(x: float, y: float) -> None:
    grid = occupancy.from_floor_plan(PLAN)
    row, col = grid.world_to_cell(x, y)
    if 0 <= row < grid.occupied.shape[0] and 0 <= col < grid.occupied.shape[1]:
        cx, cy = grid.cell_centre(row, col)
        assert abs(cx - x) <= RES / 2 + 1e-9
        assert abs(cy - y) <= RES / 2 + 1e-9


def test_pgm_has_the_top_of_the_map_first() -> None:
    grid = occupancy.from_floor_plan(PLAN)
    image = occupancy.to_image(grid)
    assert image.dtype == np.uint8
    assert image.shape == grid.occupied.shape
    # Grid row 0 is the smallest y; in the image it is the last row.
    np.testing.assert_array_equal(image[::-1] == 0, grid.occupied)
    assert set(np.unique(image)) <= {0, 254}


def test_save_writes_a_map_server_pair(tmp_path: pathlib.Path) -> None:
    grid = occupancy.from_floor_plan(PLAN)
    yaml_path = occupancy.save(grid, tmp_path / "floor")
    meta = _read_yaml(yaml_path.read_text())
    assert meta["image"] == "floor.pgm"  # relative to the YAML file
    assert meta["resolution"] == pytest.approx(RES)
    assert meta["origin"] == pytest.approx([grid.origin_x, grid.origin_y, 0.0])
    assert meta["mode"] == "trinary"
    assert meta["negate"] == 0

    pgm = (tmp_path / "floor.pgm").read_bytes()
    rows, cols = grid.occupied.shape
    header = f"P5\n{cols} {rows}\n255\n".encode()
    assert pgm.startswith(header)
    pixels = np.frombuffer(pgm[len(header) :], dtype=np.uint8)
    np.testing.assert_array_equal(
        pixels.reshape(rows, cols), occupancy.to_image(grid)
    )
