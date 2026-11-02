"""Validation of the nadir correction and its compute cost (analysis only).

For each sampled frame k (with neighbours k-1, k+1, k+2):

Position of the drone, compared with the CSV position, from
- the image center (no correction),
- the nadir pixel from CSV tilt (Omega/Kappa: the sensor-based correction),
- the nadir pixel from vision tilt, in several variants that trade compute for accuracy.

Each correction is scored raw and with a constant camera offset (pixels) calibrated on the OTHER
flights only. Tilt is also compared with the CSV directly, in degrees.

Height for speed comes from the map-match scale (focal x ground pixel size), not the CSV.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import cv2

from .dataset import Flight
from .dem import terrain_gradient, usable_agl
from .geo import haversine_m
from .motion import (
    FOCAL_PX, combine_normals, detect, estimate_from_features, intrinsics, nadir_from_normal, to_gray,
    track_pair,
)
from .repeatability import aligned_frames

# name -> (detector, work width, which neighbours)
VARIANTS = {
    "sift800": ("sift", 800, ("+1",)),
    "sift1200": ("sift", 1200, ("+1",)),
    "sift1600": ("sift", 1600, ("+1",)),
    "orb1600": ("orb", 1600, ("+1",)),
    "orb2400": ("orb", 2400, ("+1",)),
    "sift1600_k+2": ("sift", 1600, ("+2",)),
    "sift1600_both": ("sift", 1600, ("+1", "-1")),
    "sift800_both": ("sift", 800, ("+1", "-1")),
    "orb1600_both": ("orb", 1600, ("+1", "-1")),
    # Microcontroller class: gray frames, grid corners + pyramidal Lucas-Kanade, no descriptors.
    "lk320": ("lk", 320, ("+1",)),
    "lk480": ("lk", 480, ("+1",)),
    "lk640": ("lk", 640, ("+1",)),
    "lk320_both": ("lk", 320, ("+1", "-1")),
    "lk480_both": ("lk", 480, ("+1", "-1")),
}
# Variants that also get the terrain-slope correction (plane normal -> true vertical).
TERRAIN_VARIANTS = ("sift1600_both", "lk320_both", "lk480_both")


@dataclass
class FrameResult:
    flight_id: str
    frame: str
    gsd_m: float
    height_csv: float
    omega: float
    kappa: float
    resid_center_px: list[float]
    resid_csv_px: list[float]
    variants: dict = field(default_factory=dict)  # name -> {resid_px, tilt_x, tilt_y, ok, ms}
    speed_gt: float | None = None
    speed_map_scale: float | None = None  # baseline ratio x (focal x ground pixel size) / dt
    agl: float | None = None  # height above ground (CSV altitude - DEM), analysis only
    slope_deg: float | None = None  # DEM terrain slope over the footprint
    resid_csv_terrain_px: list[float] | None = None


def _nadir_csv(omega: float, kappa: float, K: np.ndarray) -> np.ndarray:
    # Milestone-2/VO fits: vision tilt_x ~ +Omega, tilt_y ~ +Kappa (+ mount offset).
    n = np.array([math.tan(math.radians(omega)), math.tan(math.radians(kappa)), 1.0])
    return nadir_from_normal(n / np.linalg.norm(n), K)


def evaluate_flight(flight: Flight, step: int, cache_dir: Path, progress: bool = True) -> list[FrameResult]:
    frames = list(flight.frames())
    idx = {f.image_path.name: i for i, f in enumerate(frames)}
    sat = flight.satellite_map()
    out = []
    for fk, al in aligned_frames(flight, step, cache_dir, progress=False):
        i = idx[fk.image_path.name]
        if i - 1 < 0 or i + 2 >= len(frames):
            continue
        nb = {"-1": frames[i - 1], "0": fk, "+1": frames[i + 1], "+2": frames[i + 2]}
        times = [datetime.fromisoformat(nb[k].date) for k in ("-1", "0", "+1", "+2")]
        if any(not (0 < (b - a).total_seconds() <= 15) for a, b in zip(times, times[1:])):
            continue
        imgs = {k: f.load_image() for k, f in nb.items()}
        h_img, w_img = imgs["0"].shape[:2]
        K = intrinsics(w_img, h_img)

        r, c = sat.latlon_to_pixel(fk.truth.lat, fk.truth.lon)
        p = np.linalg.inv(al.H) @ np.array([c, r, 1.0])
        truth_px = p[:2] / p[2]
        center = np.array([w_img / 2, h_img / 2])
        s = fk.sensors
        res = FrameResult(
            flight.flight_id, fk.image_path.name, al.scale * sat.gsd_m(), s["height"], s["Omega"], s["Kappa"],
            (truth_px - center).tolist(), (truth_px - _nadir_csv(s["Omega"], s["Kappa"], K)).tolist(),
        )

        feats: dict[tuple, tuple] = {}  # (detector, width, key) -> (Features, ms)

        def feat(det, width, key):
            if (det, width, key) not in feats:
                t = time.perf_counter()
                f = detect(imgs[key], width, det)
                feats[(det, width, key)] = (f, 1000 * (time.perf_counter() - t))
            return feats[(det, width, key)]

        grays: dict[tuple, tuple] = {}

        def gray(width, key):
            if (width, key) not in grays:
                grays[(width, key)] = to_gray(imgs[key], width)  # camera ISP output: not timed
            return grays[(width, key)]

        for name, (det, width, nbrs) in VARIANTS.items():
            if det == "lk":
                f0, ms_det = None, 0.0
            else:
                f0, ms_det = feat(det, width, "0")
            normals, ms_pairs, ok, extra = [], 0.0, True, {}
            for key in nbrs:
                t = time.perf_counter()
                if det == "lk":
                    g0, sc = gray(width, "0")
                    g1, _ = gray(width, key)
                    t = time.perf_counter()
                    m = track_pair(g0, g1, sc, K)
                else:
                    f1, _ = feat(det, width, key)
                    t = time.perf_counter()
                    m = estimate_from_features(f0, f1, K)
                ms_pairs += 1000 * (time.perf_counter() - t)
                if not m.ok:
                    ok = False
                    break
                normals.append(m.n)
                if key == "+1":
                    extra["baseline_ratio"] = m.baseline_ratio
            v = {"ok": ok, "ms_detect": ms_det, "ms_pairs": ms_pairs}
            if ok:
                n = combine_normals(normals)
                v["resid_px"] = (truth_px - nadir_from_normal(n, K)).tolist()
                v["tilt_x"] = math.degrees(math.atan2(n[0], n[2]))
                v["tilt_y"] = math.degrees(math.atan2(n[1], n[2]))
                v.update(extra)
            res.variants[name] = v

        # Terrain: the camera measures the ground plane's normal; on a slope the true vertical is
        # tilted from it by the slope. The plane-normal nadir sits uphill of the true nadir by
        # about focal x (terrain gradient in image axes). Gradient from the DEM over the image
        # footprint; image axes in the world from the map-match rotation.
        try:
            agl = usable_agl(fk) or s["height"]  # aligned frames always have a usable height
            ge, gn = terrain_gradient(fk.truth.lat, fk.truth.lon, radius_m=max(60.0, agl / 2))
            rot = math.radians(al.rotation_deg)
            ex = np.array([math.cos(rot), -math.sin(rot)])  # image +x in (east, north)
            ey = np.array([-math.sin(rot), -math.cos(rot)])  # image +y in (east, north)
            g = np.array([ge, gn])
            shift = K[0, 0] * np.array([g @ ex, g @ ey])  # pixels, nadir_normal - nadir_vertical
            res.agl, res.slope_deg = float(agl), math.degrees(math.atan(math.hypot(ge, gn)))
            for name in TERRAIN_VARIANTS:
                v = res.variants[name]
                tv = {"ok": v["ok"], "ms_detect": v["ms_detect"], "ms_pairs": v["ms_pairs"]}
                if v["ok"]:
                    tv["resid_px"] = (np.array(v["resid_px"]) + shift).tolist()
                    tv["tilt_x"], tv["tilt_y"] = v["tilt_x"], v["tilt_y"]
                res.variants[name + "+terrain"] = tv
            res.resid_csv_terrain_px = None
        except FileNotFoundError:
            pass

        dt =(times[2] - times[1]).total_seconds()
        res.speed_gt = haversine_m(fk.truth.lat, fk.truth.lon, nb["+1"].truth.lat, nb["+1"].truth.lon) / dt
        b = res.variants["sift1600"].get("baseline_ratio")
        if b is not None:
            res.speed_map_scale = b * FOCAL_PX * res.gsd_m / dt
        out.append(res)
        if progress:
            v = res.variants["sift1600"]
            err = np.hypot(*v["resid_px"]) * res.gsd_m if v["ok"] else float("nan")
            print(f"  {res.frame}: center {np.hypot(*res.resid_center_px) * res.gsd_m:.1f} m, "
                  f"sift1600 nadir {err:.1f} m", flush=True)
    return out


# --- summary --------------------------------------------------------------------------------
def _errors(rows: list[FrameResult], get, bias=np.zeros(2)) -> np.ndarray:
    e = []
    for r in rows:
        px = get(r)
        if px is not None:
            e.append(np.hypot(*(np.asarray(px) - bias)) * r.gsd_m)
    return np.array(e)


def _stats(e: np.ndarray) -> dict:
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "median_m": float(np.median(e)), "p90_m": float(np.percentile(e, 90)),
            "mean_m": float(np.mean(e))}


def summarize(rows_by_flight: dict[str, list[FrameResult]]) -> dict:
    out = {}
    flights = list(rows_by_flight)

    def getter(kind):
        if kind == "center":
            return lambda r: r.resid_center_px
        if kind == "csv":
            return lambda r: r.resid_csv_px
        return lambda r: r.variants[kind]["resid_px"] if r.variants.get(kind, {}).get("ok") else None

    kinds = ["center", "csv", *VARIANTS, *(f"{n}+terrain" for n in TERRAIN_VARIANTS)]
    for f in flights:
        rows = rows_by_flight[f]
        others = [r for g in flights if g != f for r in rows_by_flight[g]]
        s = {"frames": len(rows), "methods": {}}
        for kind in kinds:
            get = getter(kind)
            entry = {"raw": _stats(_errors(rows, get))}
            if kind != "center" and others:
                pts = [get(r) for r in others if get(r) is not None]
                if pts:
                    bias = np.median(np.array(pts), axis=0)
                    entry["calibrated"] = _stats(_errors(rows, get, bias))
                    entry["bias_px"] = bias.tolist()
            if kind not in ("center", "csv"):
                vs = [r.variants[kind] for r in rows if kind in r.variants]
                if not vs:
                    s["methods"][kind] = entry
                    continue
                entry["availability"] = float(np.mean([v["ok"] for v in vs])) if vs else 0
                entry["ms_detect"] = float(np.mean([v["ms_detect"] for v in vs])) if vs else 0
                entry["ms_pairs"] = float(np.mean([v["ms_pairs"] for v in vs])) if vs else 0
                # Tilt vs CSV, degrees. Offset calibrated on other flights (mount angle).
                ok = [(r, r.variants[kind]) for r in rows if r.variants.get(kind, {}).get("ok")]
                ok_o = [(r, r.variants[kind]) for r in others if r.variants.get(kind, {}).get("ok")]
                if ok:
                    dx = np.array([v["tilt_x"] - r.omega for r, v in ok])
                    dy = np.array([v["tilt_y"] - r.kappa for r, v in ok])
                    ox = np.median([v["tilt_x"] - r.omega for r, v in ok_o]) if ok_o else 0.0
                    oy = np.median([v["tilt_y"] - r.kappa for r, v in ok_o]) if ok_o else 0.0
                    entry["tilt_vs_csv_deg"] = {
                        "x_mean": float(dx.mean()), "x_sd": float(dx.std()),
                        "y_mean": float(dy.mean()), "y_sd": float(dy.std()),
                        "abs_median_after_offset": float(np.median(np.hypot(dx - ox, dy - oy))),
                    }
            s["methods"][kind] = entry
        slopes = [r.slope_deg for r in rows if r.slope_deg is not None]
        if slopes:
            s["slope_deg_median"] = float(np.median(slopes))
            s["slope_deg_p90"] = float(np.percentile(slopes, 90))
            s["agl_median"] = float(np.median([r.agl for r in rows if r.agl]))
        sp = [(r.speed_map_scale, r.speed_gt) for r in rows if r.speed_map_scale]
        if sp:
            e = np.array([a / b - 1 for a, b in sp])
            s["speed_from_map_scale_error_pct"] = {"median_abs": float(100 * np.median(np.abs(e))),
                                                   "bias": float(100 * np.median(e))}
        out[f] = s
    return out


def format_summary(summary: dict) -> str:
    lines = []
    for f, s in summary.items():
        terrain = (f", height above ground {s['agl_median']:.0f} m, slope median {s['slope_deg_median']:.1f} deg "
                   f"(p90 {s['slope_deg_p90']:.1f})" if "slope_deg_median" in s else "")
        lines.append(f"flight {f}: {s['frames']} frames{terrain}")
        lines.append(f"  {'method':16} {'avail':>6} {'raw med':>8} {'raw p90':>8} {'cal med':>8} {'cal p90':>8} "
                     f"{'tilt err':>8} {'ms/fix':>7}")
        for kind, e in s["methods"].items():
            raw, cal = e["raw"], e.get("calibrated", {})
            tilt = e.get("tilt_vs_csv_deg", {}).get("abs_median_after_offset")
            ms = e.get("ms_detect", 0) + e.get("ms_pairs", 0)
            lines.append(
                f"  {kind:16} {e.get('availability', 1):6.0%} {raw.get('median_m', float('nan')):8.1f} "
                f"{raw.get('p90_m', float('nan')):8.1f} {cal.get('median_m', float('nan')):8.1f} "
                f"{cal.get('p90_m', float('nan')):8.1f} "
                f"{(f'{tilt:.2f}' if tilt is not None else '-'):>8} {(f'{ms:.0f}' if ms else '-'):>7}"
            )
        if "speed_from_map_scale_error_pct" in s:
            sp = s["speed_from_map_scale_error_pct"]
            lines.append(f"  speed with height from map scale: median |error| {sp['median_abs']:.1f}%, "
                         f"bias {sp['bias']:+.1f}%")
    return "\n".join(lines)


def plot_residuals(rows_by_flight: dict[str, list[FrameResult]], summary: dict, method: str, path: Path) -> None:
    """Remaining error as arrows (image axes, meters) after calibration, per flight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    flights = list(rows_by_flight)
    fig, axes = plt.subplots(1, len(flights), figsize=(4.2 * len(flights), 4.4), squeeze=False)
    for ax, f in zip(axes[0], flights):
        rows = rows_by_flight[f]
        for kind, color in (("center", "#bbbbbb"), (method, "#1f6feb")):
            bias = np.array(summary[f]["methods"][kind].get("bias_px", [0, 0])) if kind != "center" else np.zeros(2)
            pts = []
            for r in rows:
                px = r.resid_center_px if kind == "center" else (
                    r.variants[kind]["resid_px"] if r.variants[kind]["ok"] else None)
                if px is not None:
                    pts.append((np.asarray(px) - bias) * r.gsd_m)
            pts = np.array(pts)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color,
                           label="image center" if kind == "center" else f"nadir ({method}, calibrated)")
        ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
        ax.set_aspect("equal"); ax.set_xlim(-60, 60); ax.set_ylim(60, -60)
        ax.set_title(f"flight {f}"); ax.set_xlabel("image x error (m)"); ax.set_ylabel("image y error (m)")
    axes[0][0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def run(flights: list[Flight], step: int, out_dir: Path, cache_dir: Path) -> dict:
    cv2.setNumThreads(1)  # time everything on one core, closer to a microcontroller
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for fl in flights:
        print(f"flight {fl.flight_id}", flush=True)
        rows[fl.flight_id] = evaluate_flight(fl, step, cache_dir)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "frames.json").write_text(json.dumps(
        {f: [r.__dict__ for r in rs] for f, rs in rows.items()}, default=float))
    plot_residuals(rows, summary, "sift1600_both", out_dir / "residuals.png")
    return summary
