"""The interface every localization method implements.

A localizer sees only the satellite map (offline, in `prepare`) and a drone image (online, in
`localize`). It never receives ground truth or sensor data; the harness enforces that by
construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from .geo import SatelliteMap


class Status(str, Enum):
    OK = "ok"
    UNSURE = "unsure"  # a position is given, but the method flags it as ambiguous
    FAILED = "failed"  # no position at all


@dataclass
class Fix:
    status: Status
    lat: float | None = None
    lon: float | None = None
    confidence: float | None = None
    heading_deg: float | None = None
    scale: float | None = None  # drone px -> map px, for altitude later
    note: str = ""

    @classmethod
    def failed(cls, note: str = "") -> Fix:
        return cls(Status.FAILED, note=note)


class Localizer(Protocol):
    name: str

    def prepare(self, sat_map: SatelliteMap) -> None:
        """Offline step: build whatever index the method needs from the map."""

    def index_size_bytes(self) -> int:
        """Size of the stored map index, the thing that would ship on the drone."""

    def localize(self, image: np.ndarray) -> Fix:
        """Online step: RGB uint8 drone image -> position."""
