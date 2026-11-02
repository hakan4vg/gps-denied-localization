from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from gpsloc.dataset import Flight
from gpsloc.evaluate import run, summarize, FrameResult
from gpsloc.geo import SatelliteMap, haversine_m, offset_latlon
from gpsloc.localizer import Fix, Status


def test_haversine_one_degree_latitude():
    assert haversine_m(0, 0, 1, 0) == pytest.approx(111_195, rel=1e-3)


def test_offset_roundtrip():
    lat, lon = offset_latlon(32.3, 119.9, north_m=100, east_m=-50)
    assert haversine_m(32.3, 119.9, lat, lon) == pytest.approx(np.hypot(100, 50), rel=1e-3)


def _write_map(path: Path, west=119.80, south=32.29, east=119.81, north=32.30, w=400, h=300):
    data = np.random.default_rng(0).integers(0, 255, (3, h, w), dtype=np.uint8)
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(west, south, east, north, w, h),
    ) as dst:
        dst.write(data)


def test_pixel_latlon_roundtrip(tmp_path):
    p = tmp_path / "m.tif"
    _write_map(p)
    m = SatelliteMap(p)
    lat, lon = m.pixel_to_latlon(120.25, 37.5)
    r, c = m.latlon_to_pixel(lat, lon)
    assert (r, c) == pytest.approx((120.25, 37.5), abs=1e-6)
    s, w, n, e = m.bounds_latlon()
    assert s > 32.29 and n < 32.30 and w > 119.80 and e < 119.81
    assert m.read(max_size=100).shape == (75, 100, 3)


def _make_flight(root: Path) -> Flight:
    d = root / "07"
    (d / "drone").mkdir(parents=True)
    _write_map(d / "satellite07.tif")
    import cv2

    rows = ["num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2"]
    for i, (lat, lon) in enumerate([(32.295, 119.805), (32.2951, 119.8051)], 1):
        name = f"07_{i:04d}.JPG"
        cv2.imwrite(str(d / "drone" / name), np.zeros((20, 30, 3), np.uint8))
        rows.append(f"{i},{name},2020-01-01,{lat},{lon},400,0,0,10,10")
    (d / "07.csv").write_text("\n".join(rows) + "\n")
    return Flight(root, "07")


class _Truthful:
    """Test double that cheats by reading a fixed answer list; checks the harness, not a method."""

    name = "fixed"

    def __init__(self, answers):
        self.answers = list(answers)

    def prepare(self, sat_map):
        pass

    def index_size_bytes(self):
        return 123

    def localize(self, image):
        assert image.ndim == 3
        return self.answers.pop(0)


def test_run_scores_and_counts_failures(tmp_path):
    flight = _make_flight(tmp_path)
    near = offset_latlon(32.295, 119.805, 30, 0)  # 30 m off
    answers = [Fix(Status.OK, *near), Fix.failed("no match")]
    summary, results = run(_Truthful(answers), flight, out_dir=tmp_path / "out", progress=False)
    assert results[0].error_m == pytest.approx(30, rel=1e-2)
    assert results[1].error_m is None
    assert summary.success_rate == {"25m": 0.0, "50m": 0.5, "100m": 0.5}
    assert summary.n_failed == 1 and summary.index_size_bytes == 123
    assert (tmp_path / "out" / "summary.json").exists()


def test_unsure_counts_toward_overall_but_not_ok_only():
    base = dict(flight_id="x", seconds=0.1, confidence=None, pred_lat=0, pred_lon=0, true_lat=0, true_lon=0)
    results = [
        FrameResult(filename="a", status="ok", error_m=10, **base),
        FrameResult(filename="b", status="unsure", error_m=10, **base),
    ]
    s = summarize(results, "m", "x", 0, 0)
    assert s.success_rate["25m"] == 1.0
    assert s.success_rate_ok_only["25m"] == 0.5
