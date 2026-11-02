"""Geographic helpers: distances, GeoTIFF pixel <-> WGS84, and windowed map reads.

Adapted from the v2 helpers in localize_roads.py (_pixel_to_wgs84, _wgs84_to_dataset,
_estimate_map_gsd_mpp, _read_satellite_crop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

EARTH_RADIUS_M = 6_371_008.8
WGS84 = "EPSG:4326"


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Move a point by a small north/east offset in meters (local flat-earth approximation)."""
    dlat = math.degrees(north_m / EARTH_RADIUS_M)
    dlon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


@dataclass
class SatelliteMap:
    """A georeferenced satellite map. Opens the file lazily; never loads it whole by default."""

    path: Path

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        with rasterio.open(self.path) as src:
            self.width: int = src.width
            self.height: int = src.height
            self.crs = src.crs
            self.transform: Affine = src.transform
            self.band_count: int = src.count

    # --- coordinate conversion -------------------------------------------------
    def pixel_to_latlon(self, row: float, col: float) -> tuple[float, float]:
        """Pixel (row, col), measured to pixel centers, -> (lat, lon)."""
        x, y = self.transform @ (col + 0.5, row + 0.5)
        if self.crs is None or self.crs.to_string() == WGS84:
            return float(y), float(x)
        lons, lats = warp_transform(self.crs, WGS84, [x], [y])
        return float(lats[0]), float(lons[0])

    def latlon_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        """(lat, lon) -> fractional pixel (row, col), inverse of pixel_to_latlon."""
        if self.crs is None or self.crs.to_string() == WGS84:
            x, y = lon, lat
        else:
            xs, ys = warp_transform(WGS84, self.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        col, row = ~self.transform @ (x, y)
        return float(row) - 0.5, float(col) - 0.5

    def bounds_latlon(self) -> tuple[float, float, float, float]:
        """(south, west, north, east) of the map's corner pixels."""
        corners = [
            self.pixel_to_latlon(r, c)
            for r, c in ((0, 0), (0, self.width - 1), (self.height - 1, 0), (self.height - 1, self.width - 1))
        ]
        lats = [p[0] for p in corners]
        lons = [p[1] for p in corners]
        return min(lats), min(lons), max(lats), max(lons)

    def center_latlon(self) -> tuple[float, float]:
        return self.pixel_to_latlon((self.height - 1) / 2, (self.width - 1) / 2)

    def gsd_m(self, row: float | None = None, col: float | None = None) -> float:
        """Ground size of one pixel in meters (mean of x and y), measured at a pixel."""
        row = self.height / 2 if row is None else row
        col = self.width / 2 if col is None else col
        lat0, lon0 = self.pixel_to_latlon(row, col)
        lat1, lon1 = self.pixel_to_latlon(row, col + 1)
        lat2, lon2 = self.pixel_to_latlon(row + 1, col)
        return (haversine_m(lat0, lon0, lat1, lon1) + haversine_m(lat0, lon0, lat2, lon2)) / 2

    # --- reading ---------------------------------------------------------------
    def read(
        self,
        row: int = 0,
        col: int = 0,
        height: int | None = None,
        width: int | None = None,
        max_size: int | None = None,
    ) -> np.ndarray:
        """Read an RGB window as HxWx3 uint8, optionally downscaled so its long side <= max_size."""
        height = self.height - row if height is None else height
        width = self.width - col if width is None else width
        win = Window(col, row, width, height)
        out_shape = None
        if max_size is not None and max(height, width) > max_size:
            s = max_size / max(height, width)
            out_shape = (min(3, self.band_count), max(1, round(height * s)), max(1, round(width * s)))
        with rasterio.open(self.path) as src:
            bands = list(range(1, min(3, src.count) + 1))
            data = src.read(
                bands, window=win, out_shape=out_shape, resampling=Resampling.average, boundless=False
            )
        if data.dtype != np.uint8:
            data = np.clip(data, 0, 255).astype(np.uint8)
        img = np.transpose(data, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        return img
