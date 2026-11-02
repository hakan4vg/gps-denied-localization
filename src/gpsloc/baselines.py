"""Trivial localizers. They set the floor any real method has to beat, and test the harness."""

from __future__ import annotations

import numpy as np

from .geo import SatelliteMap
from .localizer import Fix, Status


class MapCenter:
    """Always answers the center of the map."""

    name = "map-center"

    def prepare(self, sat_map: SatelliteMap) -> None:
        self._center = sat_map.center_latlon()

    def index_size_bytes(self) -> int:
        return 16

    def localize(self, image: np.ndarray) -> Fix:
        lat, lon = self._center
        return Fix(Status.OK, lat, lon, confidence=0.0)


class RandomPoint:
    """Answers a uniformly random pixel of the map (seeded, so runs repeat)."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def prepare(self, sat_map: SatelliteMap) -> None:
        self._map = sat_map

    def index_size_bytes(self) -> int:
        return 0

    def localize(self, image: np.ndarray) -> Fix:
        row = self._rng.uniform(0, self._map.height - 1)
        col = self._rng.uniform(0, self._map.width - 1)
        lat, lon = self._map.pixel_to_latlon(row, col)
        return Fix(Status.OK, lat, lon, confidence=0.0)


REGISTRY = {"map-center": MapCenter, "random": RandomPoint}
