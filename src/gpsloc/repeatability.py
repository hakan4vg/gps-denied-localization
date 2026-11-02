"""Landmark repeatability between drone and satellite images (milestone 2 go/no-go).

For each aligned frame:
1. Resample the drone image and the map patch under it to the same ground resolution.
2. Run the same extractor on both.
3. Project drone points onto the map with the reference alignment (analysis only).
4. Count drone points with a map point within r meters, and compare with chance: how often a
   random point would have a map point that close, given the map point density.

Quad hashing needs all 4 points of a quad to repeat, so rate**4 is the rough share of usable
quads. A useful extractor needs a high rate well above chance AND enough points per image.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .align import Alignment, align_frame
from .dataset import Flight, Frame
from .geo import SatelliteMap
from .landmarks import EXTRACTORS

RADII_M = (3.0, 5.0, 10.0)


# --- alignment cache ------------------------------------------------------------------------
def load_alignments(cache: Path) -> dict[str, dict]:
    return json.loads(cache.read_text()) if cache.exists() else {}


def aligned_frames(flight: Flight, step: int, cache_dir: Path, progress: bool = True):
    """Yield (frame, Alignment) for frames that aligned, computing and caching as needed."""
    cache = Path(cache_dir) / f"align_{flight.flight_id}.json"
    data = load_alignments(cache)
    sat = flight.satellite_map()
    dirty = False
    for k, fr in enumerate(flight.frames(step=step)):
        name = fr.image_path.name
        if name not in data:
            a = align_frame(fr, sat)
            data[name] = {
                "ok": a.ok, "H": a.H.tolist() if a.H is not None else None, "inliers": a.inliers,
                "scale": a.scale, "rotation_deg": a.rotation_deg, "center_offset_m": a.center_offset_m,
                "center_offset_ne": a.center_offset_ne, "note": a.note,
            }
            dirty = True
            if progress and k % 10 == 0:
                print(f"  aligned {name}: {'ok' if a.ok else a.note}", flush=True)
        d = data[name]
        if d["ok"]:
            yield fr, Alignment(True, np.array(d["H"]), d["inliers"], d["scale"], d["rotation_deg"],
                                d["center_offset_m"], tuple(d["center_offset_ne"]))
    if dirty:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))


# --- per-frame measurement ------------------------------------------------------------------
@dataclass
class FrameRepeat:
    filename: str
    n_drone: int
    n_map: int
    area_km2: float
    rate: dict[str, float]  # share of drone points with a map point within r
    rate_map: dict[str, float]  # share of map points with a drone point within r
    chance: dict[str, float]
    seconds_drone: float
    seconds_map: float


def _views(frame: Frame, al: Alignment, sat: SatelliteMap, gsd: float, image: np.ndarray | None = None):
    """Drone and map patch at `gsd` m/px, and the 3x3 transform drone-work -> map-work."""
    img = frame.load_image() if image is None else image
    gsd_map = sat.gsd_m()
    gsd_drone = al.scale * gsd_map
    kd, km = gsd_drone / gsd, gsd_map / gsd
    drone = cv2.resize(img, None, fx=kd, fy=kd, interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    corners = al.drone_to_map(np.array([[0, 0], [w, 0], [w, h], [0, h]], float))
    pad = 20.0 / gsd_map
    c0 = int(max(0, corners[:, 0].min() - pad)); c1 = int(min(sat.width, corners[:, 0].max() + pad))
    r0 = int(max(0, corners[:, 1].min() - pad)); r1 = int(min(sat.height, corners[:, 1].max() + pad))
    patch = sat.read(r0, c0, r1 - r0, c1 - c0, max_size=max(1, round(max(r1 - r0, c1 - c0) * km)))
    sx, sy = patch.shape[1] / (c1 - c0), patch.shape[0] / (r1 - r0)
    M = np.array([[sx, 0, -c0 * sx], [0, sy, -r0 * sy], [0, 0, 1]])
    D_inv = np.diag([1 / kd, 1 / kd, 1.0])
    T = M @ al.H @ D_inv
    return drone, patch, T / T[2, 2]


def _project(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return pts.reshape(0, 2)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), T).reshape(-1, 2)


def measure_frame(
    frame: Frame, al: Alignment, sat: SatelliteMap, extractor: str, gsd: float = 1.0,
    margin_m: float = 15.0, image: np.ndarray | None = None,
):
    extract = EXTRACTORS[extractor]
    drone, patch, T = _views(frame, al, sat, gsd, image)
    t = time.perf_counter(); pd = extract(drone, gsd); td = time.perf_counter() - t
    t = time.perf_counter(); pm = extract(patch, gsd); tm = time.perf_counter() - t

    m = margin_m / gsd
    hd, wd = drone.shape[:2]

    def inside_drone(p):  # p in drone-work coords
        return (p[:, 0] >= m) & (p[:, 0] < wd - m) & (p[:, 1] >= m) & (p[:, 1] < hd - m)

    pd_in = pd[inside_drone(pd)] if len(pd) else pd.reshape(0, 2)
    pd_map = _project(pd_in, T)
    pm_in = pm[inside_drone(_project(pm, np.linalg.inv(T)))] if len(pm) else pm.reshape(0, 2)
    area_km2 = (wd - 2 * m) * (hd - 2 * m) * gsd * gsd / 1e6

    rate, rate_map, chance = {}, {}, {}
    density = len(pm_in) / (area_km2 * 1e6) if area_km2 > 0 else 0.0
    d_nn = cKDTree(pm_in).query(pd_map)[0] * gsd if len(pm_in) and len(pd_map) else np.array([])
    m_nn = cKDTree(pd_map).query(pm_in)[0] * gsd if len(pm_in) and len(pd_map) else np.array([])
    for r in RADII_M:
        key = f"{r:g}m"
        rate[key] = float(np.mean(d_nn <= r)) if len(d_nn) else 0.0
        rate_map[key] = float(np.mean(m_nn <= r)) if len(m_nn) else 0.0
        chance[key] = 1.0 - math.exp(-density * math.pi * r * r)
    result = FrameRepeat(frame.image_path.name, len(pd_in), len(pm_in), area_km2, rate, rate_map,
                         chance, td, tm)
    return result, (drone, patch, T, pd_in, pd_map, pm_in)


def draw(drone, patch, T, pd_map, pm_in, gsd: float, r_m: float = 5.0) -> np.ndarray:
    """Side by side: map patch | drone warped onto the map. Blue = map points, green/red =
    drone points with / without a map point within r_m."""
    h, w = patch.shape[:2]
    warped = cv2.warpPerspective(drone, T, (w, h))
    left = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
    near = (cKDTree(pm_in).query(pd_map)[0] * gsd <= r_m) if len(pm_in) and len(pd_map) else np.zeros(len(pd_map), bool)
    for img in (left, right):
        for x, y in pm_in:
            cv2.circle(img, (int(x), int(y)), 5, (255, 120, 0), 1, cv2.LINE_AA)
        for (x, y), ok in zip(pd_map, near):
            cv2.drawMarker(img, (int(x), int(y)), (0, 220, 0) if ok else (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
    sep = np.full((h, 4, 3), 255, np.uint8)
    return np.hstack([left, sep, right])


def summarize(results: list[FrameRepeat], extractor: str, flight_id: str, gsd: float) -> dict:
    def wmean(key: str, field: str) -> float:
        # Pooled over points, so frames with many points weigh more.
        num = sum(getattr(r, field)[key] * (r.n_drone if field != "rate_map" else r.n_map) for r in results)
        den = sum((r.n_drone if field != "rate_map" else r.n_map) for r in results)
        return num / den if den else 0.0

    n_d = np.array([r.n_drone for r in results])
    area = np.array([r.area_km2 for r in results])
    out = {
        "extractor": extractor, "flight_id": flight_id, "gsd_m": gsd, "frames": len(results),
        "drone_points_per_frame_median": float(np.median(n_d)) if len(n_d) else 0,
        "map_points_per_frame_median": float(np.median([r.n_map for r in results])) if results else 0,
        "drone_points_per_km2": float(n_d.sum() / area.sum()) if area.sum() else 0,
        "footprint_km2_median": float(np.median(area)) if len(area) else 0,
        "ms_per_drone_image": 1000 * float(np.mean([r.seconds_drone for r in results])) if results else 0,
    }
    for r in RADII_M:
        k = f"{r:g}m"
        rate = wmean(k, "rate")
        chance = float(np.mean([x.chance[k] for x in results])) if results else 0
        out[f"rate_{k}"] = rate
        out[f"rate_map_{k}"] = wmean(k, "rate_map")
        out[f"chance_{k}"] = chance
        out[f"quad_survival_{k}"] = rate**4
    return out


def format_summary(s: dict) -> str:
    lines = [
        f"{s['extractor']} on flight {s['flight_id']} at {s['gsd_m']} m/px: {s['frames']} frames, "
        f"median {s['drone_points_per_frame_median']:.0f} drone / {s['map_points_per_frame_median']:.0f} map points "
        f"per frame ({s['drone_points_per_km2']:.0f}/km2, footprint {s['footprint_km2_median']:.3f} km2), "
        f"{s['ms_per_drone_image']:.0f} ms/image"
    ]
    for r in RADII_M:
        k = f"{r:g}m"
        lines.append(
            f"  within {k:>4}: drone->map {s[f'rate_{k}']:6.1%}  map->drone {s[f'rate_map_{k}']:6.1%}  "
            f"chance {s[f'chance_{k}']:6.1%}  all-4-of-a-quad {s[f'quad_survival_{k}']:6.2%}"
        )
    return "\n".join(lines)


def run(flight: Flight, extractor: str, step: int, gsd: float, out_dir: Path, cache_dir: Path,
        n_viz: int = 4, limit: int | None = None) -> dict:
    sat = flight.satellite_map()
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for k, (fr, al) in enumerate(aligned_frames(flight, step, cache_dir)):
        if limit is not None and k >= limit:
            break
        res, views = measure_frame(fr, al, sat, extractor, gsd)
        results.append(res)
        if k < n_viz:
            drone, patch, T, _, pd_map, pm_in = views
            cv2.imwrite(str(out_dir / f"viz_{fr.image_path.stem}.jpg"),
                        draw(drone, patch, T, pd_map, pm_in, gsd), [cv2.IMWRITE_JPEG_QUALITY, 85])
    summary = summarize(results, extractor, flight.flight_id, gsd)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / "frames.json").open("w") as f:
        json.dump([asdict(r) for r in results], f)
    return summary
