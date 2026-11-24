"""Turns a floor plan into an occupancy grid for Nav2's map_server.

The grid is the map Nav2 plans on (R8). It holds walls only: furniture and
people are what the camera has to find (R10, M4).

Layout: occupied[row, col] covers x in [origin_x + col * res, + res) and
y in [origin_y + row * res, + res), so row 0 is the smallest y. map_server
images run the other way (first row = largest y); to_image flips them.
"""

import dataclasses
import json
import math
import pathlib
from collections.abc import Iterable

import numpy as np

from minifsd.world import floor_plan

RESOLUTION = 0.05  # metres per cell, Nav2's usual value
MARGIN = 0.5  # free border around the walls, metres

# Pixel values in trinary mode with negate 0: map_server reads
# p = (255 - value) / 255 and calls p > 0.65 occupied, p < 0.25 free.
OCCUPIED_PIXEL = 0
FREE_PIXEL = 254

# Cells this close to touching a wall edge (in cells) don't count, so
# rounding error never grows a wall by a whole cell.
_EPS = 1e-6


@dataclasses.dataclass(frozen=True, eq=False)
class Grid:
    """A 2D occupancy grid in the world (odom/map) frame."""

    occupied: np.ndarray  # (rows, cols) bool
    origin_x: float  # world x of the corner of cell (0, 0)
    origin_y: float
    resolution: float = RESOLUTION

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """The (row, col) of the cell holding a world point."""
        col = math.floor((x - self.origin_x) / self.resolution)
        row = math.floor((y - self.origin_y) / self.resolution)
        return row, col

    def cell_centre(self, row: int, col: int) -> tuple[float, float]:
        """The world (x, y) of a cell's centre."""
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )


def rasterize(
    walls: Iterable[floor_plan.Wall],
    resolution: float = RESOLUTION,
    margin: float = MARGIN,
) -> Grid:
    """Marks every cell that overlaps a wall as occupied.

    Args:
        walls: Axis-aligned wall rectangles.
        resolution: Cell size, metres.
        margin: Free border around the walls' bounding box, metres.

    Returns:
        The grid. Its origin is a whole number of cells, so walls on round
        coordinates line up with cell edges.
    """
    walls = list(walls)
    x0 = min(w.x_min for w in walls) - margin
    y0 = min(w.y_min for w in walls) - margin
    x1 = max(w.x_max for w in walls) + margin
    y1 = max(w.y_max for w in walls) + margin
    # round() drops float noise such as -0.6000000000000001.
    origin_x = round(math.floor(x0 / resolution) * resolution, 9)
    origin_y = round(math.floor(y0 / resolution) * resolution, 9)
    cols = math.ceil((x1 - origin_x) / resolution)
    rows = math.ceil((y1 - origin_y) / resolution)
    occupied = np.zeros((rows, cols), dtype=bool)

    def span(lo: float, hi: float, origin: float) -> slice:
        # Cells whose interval overlaps (lo, hi) by more than a touch.
        first = math.floor((lo - origin) / resolution + _EPS)
        stop = math.ceil((hi - origin) / resolution - _EPS)
        return slice(first, stop)

    for w in walls:
        occupied[
            span(w.y_min, w.y_max, origin_y), span(w.x_min, w.x_max, origin_x)
        ] = True
    return Grid(occupied, origin_x, origin_y, resolution)


def from_floor_plan(plan: floor_plan.FloorPlan) -> Grid:
    """The occupancy grid of a floor plan's walls."""
    return rasterize(plan.walls)


def to_image(grid: Grid) -> np.ndarray:
    """The grid as a map_server image: uint8, first row = largest y."""
    image = np.where(grid.occupied, OCCUPIED_PIXEL, FREE_PIXEL)
    return image[::-1].astype(np.uint8)


def save(grid: Grid, stem: pathlib.Path) -> pathlib.Path:
    """Writes stem.pgm and stem.yaml, the pair map_server loads.

    Args:
        grid: The grid to save.
        stem: Path without an extension, e.g. maps/floor_0.

    Returns:
        The YAML path, which is what map_server's yaml_filename takes.
    """
    image = to_image(grid)
    rows, cols = image.shape
    pgm = stem.with_suffix(".pgm")
    # Binary PGM (P5): a text header, then one byte per pixel.
    pgm.write_bytes(f"P5\n{cols} {rows}\n255\n".encode() + image.tobytes())

    meta = {
        "image": pgm.name,  # relative to the YAML file
        "mode": "trinary",
        "resolution": grid.resolution,
        "origin": [grid.origin_x, grid.origin_y, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
    }
    yaml_path = stem.with_suffix(".yaml")
    # JSON values are valid YAML, which keeps PyYAML out of the core env.
    lines = [
        f"{k}: {v if isinstance(v, str) else json.dumps(v)}"
        for k, v in meta.items()
    ]
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path
