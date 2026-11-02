"""UAV-VisLoc loader.

Layout (checked against the download, 2026-11-02):

    UAV-VisLoc/
      satellite_ coordinates_range.csv   # note the space in the filename
      03/
        03.csv          # num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2
        satellite03.tif # EPSG:4326 GeoTIFF
        drone/03_0001.JPG ...

lat/lon in the CSV are ground truth: only the scoring harness reads them. height and the
angles are sensor data: kept in `Frame.sensors` for analysis, never passed to a localizer.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import SatelliteMap

SENSOR_COLUMNS = ("height", "Omega", "Kappa", "Phi1", "Phi2")


@dataclass(frozen=True)
class GroundTruth:
    lat: float
    lon: float


@dataclass(frozen=True)
class Frame:
    flight_id: str
    index: int
    image_path: Path
    date: str
    truth: GroundTruth
    sensors: dict[str, float] = field(default_factory=dict)

    def load_image(self, max_size: int | None = None) -> np.ndarray:
        """RGB uint8 image, optionally downscaled so its long side <= max_size."""
        import cv2

        img = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if max_size is not None and max(img.shape[:2]) > max_size:
            s = max_size / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return img


class Flight:
    def __init__(self, root: Path | str, flight_id: str):
        self.root = Path(root)
        self.flight_id = flight_id
        self.dir = self.root / flight_id
        self.csv_path = self.dir / f"{flight_id}.csv"
        self.map_path = self.dir / f"satellite{flight_id}.tif"
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        self._frames = self._read_csv()

    def _read_csv(self) -> list[Frame]:
        frames = []
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sensors = {}
                for key in SENSOR_COLUMNS:
                    value = row.get(key, "")
                    if value not in ("", None):
                        sensors[key] = float(value)
                frames.append(
                    Frame(
                        flight_id=self.flight_id,
                        index=int(row["num"]),
                        image_path=self.dir / "drone" / row["filename"],
                        date=row.get("date", ""),
                        truth=GroundTruth(float(row["lat"]), float(row["lon"])),
                        sensors=sensors,
                    )
                )
        return frames

    def __len__(self) -> int:
        return len(self._frames)

    def frames(self, step: int = 1, limit: int | None = None) -> Iterator[Frame]:
        selected = self._frames[::step]
        return iter(selected[:limit] if limit is not None else selected)

    def satellite_map(self) -> SatelliteMap:
        return SatelliteMap(self.map_path)


def available_flights(root: Path | str) -> list[str]:
    root = Path(root)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and (p / f"{p.name}.csv").exists()
    )
