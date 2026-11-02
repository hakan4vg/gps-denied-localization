"""Ground elevation from Copernicus GLO-30 DEM tiles (30 m, free, no account).

Tiles: https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N32_00_E119_00_DEM/...
Default folder: ~/data/dem (override with GPSLOC_DEM).

Note: the UAV-VisLoc CSV `height` is altitude above sea level, not above ground. On the
near-sea-level flights (03, 04) they coincide; on hilly flights (05, 06, 11) they differ by
200-2000 m. Height above ground = CSV height - DEM elevation.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

# Below this the CSV altitude is a glitch (flight 06 frames 93-98 drop ~340 m for six frames).
MIN_AGL_M = 20.0

DEFAULT_DIR = Path(os.environ.get("GPSLOC_DEM", Path.home() / "data" / "dem"))


def tile_name(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(math.floor(lat)):02d}_00_{ew}{abs(math.floor(lon)):03d}_00_DEM"


def tile_url(lat: float, lon: float) -> str:
    n = tile_name(lat, lon)
    return f"https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif"


@lru_cache(maxsize=16)
def _open(path: str):
    return rasterio.open(path)


def _src(lat: float, lon: float, dem_dir: Path):
    path = Path(dem_dir) / f"{tile_name(lat, lon)}.tif"
    if not path.exists():
        raise FileNotFoundError(f"{path} (download from {tile_url(lat, lon)})")
    return _open(str(path))


def elevation(lat: float, lon: float, dem_dir: Path = DEFAULT_DIR) -> float:
    """Bilinear ground elevation in meters."""
    src = _src(lat, lon, dem_dir)
    col, row = ~src.transform * (lon, lat)
    c0, r0 = int(math.floor(col - 0.5)), int(math.floor(row - 0.5))
    a = src.read(1, window=Window(c0, r0, 2, 2), boundless=True, fill_value=0).astype(float)
    fx, fy = col - 0.5 - c0, row - 0.5 - r0
    return float(a[0, 0] * (1 - fx) * (1 - fy) + a[0, 1] * fx * (1 - fy) + a[1, 0] * (1 - fx) * fy + a[1, 1] * fx * fy)


def slope_deg(lat: float, lon: float, radius_m: float = 150.0, dem_dir: Path = DEFAULT_DIR) -> tuple[float, float]:
    """Mean terrain slope (degrees) and downhill-facing aspect (degrees from north)."""
    ge, gn = terrain_gradient(lat, lon, radius_m, dem_dir)
    return math.degrees(math.atan(math.hypot(ge, gn))), math.degrees(math.atan2(-ge, -gn)) % 360


def terrain_gradient(lat: float, lon: float, radius_m: float = 150.0, dem_dir: Path = DEFAULT_DIR) -> tuple[float, float]:
    """(rise per meter east, rise per meter north) of a plane fitted to the DEM within radius_m:
    the ground plane a camera above this point would see."""
    src = _src(lat, lon, dem_dir)
    col, row = ~src.transform * (lon, lat)
    px_m_y = abs(src.transform.e) * 111_320.0
    px_m_x = abs(src.transform.a) * 111_320.0 * math.cos(math.radians(lat))
    rx, ry = int(math.ceil(radius_m / px_m_x)), int(math.ceil(radius_m / px_m_y))
    a = src.read(1, window=Window(int(col) - rx, int(row) - ry, 2 * rx + 1, 2 * ry + 1), boundless=True,
                 fill_value=0).astype(float)
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    east, north = xx * px_m_x, -yy * px_m_y
    inside = east**2 + north**2 <= radius_m**2
    A = np.column_stack([east[inside], north[inside], np.ones(inside.sum())])
    (ge, gn, _), *_ = np.linalg.lstsq(A, a[inside], rcond=None)
    return float(ge), float(gn)


def height_above_ground(frame, dem_dir: Path = DEFAULT_DIR) -> float | None:
    """CSV altitude minus ground elevation at the CSV position (analysis only)."""
    h = frame.sensors.get("height")
    if h is None:
        return None
    try:
        return h - elevation(frame.truth.lat, frame.truth.lon, dem_dir)
    except FileNotFoundError:
        return None


def usable_agl(frame, dem_dir: Path = DEFAULT_DIR) -> float | None:
    """Height above ground for scaling images: DEM-corrected where a tile exists, else the raw
    CSV altitude. None when missing or implausibly low (bad CSV altitude)."""
    agl = height_above_ground(frame, dem_dir)
    if agl is None:
        agl = frame.sensors.get("height")
    if agl is None or not agl >= MIN_AGL_M:
        return None
    return float(agl)
