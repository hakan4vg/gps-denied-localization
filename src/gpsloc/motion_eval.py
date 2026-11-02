"""Evaluate frame-pair motion estimates against the CSV (analysis only).

For consecutive frames (k, k+1):
- speed: baseline ratio x height / dt vs distance between CSV positions / dt
  (the CSV height only converts "heights per second" to m/s; the ratio error is the same)
- direction of travel: image-frame direction rotated into the map with the reference alignment,
  vs the bearing between CSV positions
- tilt: vision tilt about image x/y axes vs CSV Omega/Kappa
- position: map location of the nadir pixel (and of the image center) vs the CSV position.
  A constant camera offset (principal point / mount angle, in pixels) is calibrated on one
  flight and applied to the other, so no flight is scored with its own calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .dataset import Flight
from .geo import haversine_m
from .motion import estimate_pair, intrinsics
from .repeatability import aligned_frames


@dataclass
class PairRow:
    flight_id: str
    frame: str
    ok: bool
    note: str
    inliers: int
    dt_s: float
    speed_gt: float
    speed_est: float | None = None
    bearing_gt: float | None = None
    bearing_est: float | None = None
    tilt_deg: float | None = None
    tilt_x_deg: float | None = None  # rotation of the view along image x (atan n_x / n_z)
    tilt_y_deg: float | None = None
    omega: float | None = None
    kappa: float | None = None
    yaw_change_deg: float | None = None
    dphi1: float | None = None
    # residual (pixels, full res) from nadir pixel to where the CSV position falls in the image
    resid_nadir_px: tuple[float, float] | None = None
    resid_center_px: tuple[float, float] | None = None
    gsd_m: float | None = None  # meters per drone pixel at this frame (from the alignment)


def _bearing(lat1, lon1, lat2, lon2) -> float:
    north = haversine_m(lat1, lon1, lat2, lon1) * np.sign(lat2 - lat1)
    east = haversine_m(lat1, lon1, lat1, lon2) * np.sign(lon2 - lon1)
    return math.degrees(math.atan2(east, north)) % 360


def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def evaluate_flight(flight: Flight, step: int, cache_dir: Path, progress: bool = True) -> list[PairRow]:
    frames = list(flight.frames())
    by_name = {f.image_path.name: i for i, f in enumerate(frames)}
    sat = flight.satellite_map()
    rows = []
    for f1, al in aligned_frames(flight, step, cache_dir, progress=False):
        i = by_name[f1.image_path.name]
        if i + 1 >= len(frames):
            continue
        f2 = frames[i + 1]
        dt = (datetime.fromisoformat(f2.date) - datetime.fromisoformat(f1.date)).total_seconds()
        dist = haversine_m(f1.truth.lat, f1.truth.lon, f2.truth.lat, f2.truth.lon)
        row = PairRow(flight.flight_id, f1.image_path.name, False, "", 0, dt, dist / dt if dt > 0 else math.nan,
                      bearing_gt=_bearing(f1.truth.lat, f1.truth.lon, f2.truth.lat, f2.truth.lon),
                      omega=f1.sensors.get("Omega"), kappa=f1.sensors.get("Kappa"),
                      dphi1=_angle_diff(f2.sensors["Phi1"], f1.sensors["Phi1"]))
        if not (0 < dt <= 15):
            row.note = f"dt {dt:.0f}s"
            rows.append(row)
            continue
        img1, img2 = f1.load_image(), f2.load_image()
        K = intrinsics(img1.shape[1], img1.shape[0])
        m = estimate_pair(img1, img2, K)
        row.ok, row.note, row.inliers = m.ok, m.note, m.inliers
        if m.ok:
            row.speed_est = m.baseline_ratio * f1.sensors["height"] / dt
            # image direction -> map (col, row) direction -> bearing from north
            phi = math.radians(m.travel_dir_img_deg + al.rotation_deg)
            row.bearing_est = math.degrees(math.atan2(math.cos(phi), -math.sin(phi))) % 360
            row.tilt_deg = m.tilt_deg
            row.tilt_x_deg = math.degrees(math.atan2(m.n[0], m.n[2]))
            row.tilt_y_deg = math.degrees(math.atan2(m.n[1], m.n[2]))
            row.yaw_change_deg = m.yaw_change_deg
            # Where does the CSV position fall in frame k? Invert the reference alignment.
            r, c = sat.latlon_to_pixel(f1.truth.lat, f1.truth.lon)
            p = np.linalg.inv(al.H) @ np.array([c, r, 1.0])
            p = p[:2] / p[2]
            nadir = m.nadir_px(K)
            center = np.array([img1.shape[1] / 2, img1.shape[0] / 2])
            row.resid_nadir_px = tuple(map(float, p - nadir))
            row.resid_center_px = tuple(map(float, p - center))
            row.gsd_m = al.scale * sat.gsd_m()
        rows.append(row)
        if progress:
            if m.ok:
                print(f"  {row.frame}: inl {m.inliers} speed {row.speed_est:.1f}/{row.speed_gt:.1f} m/s "
                      f"tilt {row.tilt_deg:.2f} deg", flush=True)
            else:
                print(f"  {row.frame}: {m.note}", flush=True)
    return rows


def _pos_errors(rows: list[PairRow], bias_px: np.ndarray, which: str) -> np.ndarray:
    out = []
    for r in rows:
        res = np.array(getattr(r, which)) - bias_px
        out.append(np.hypot(*res) * r.gsd_m)
    return np.array(out)


def summarize(rows_by_flight: dict[str, list[PairRow]]) -> dict:
    out = {}
    ok_rows = {f: [r for r in rows if r.ok] for f, rows in rows_by_flight.items()}
    flights = list(ok_rows)
    for f in flights:
        R = ok_rows[f]
        all_rows = rows_by_flight[f]
        s = {"pairs": len(all_rows), "ok": len(R)}
        if not R:
            out[f] = s
            continue
        sp_err = np.array([(r.speed_est - r.speed_gt) / r.speed_gt for r in R])
        br_err = np.array([_angle_diff(r.bearing_est, r.bearing_gt) for r in R])
        s["speed_error_pct_median"] = float(100 * np.median(np.abs(sp_err)))
        s["speed_bias_pct"] = float(100 * np.median(sp_err))
        s["bearing_error_deg_median"] = float(np.median(np.abs(br_err)))
        tx = np.array([r.tilt_x_deg for r in R]); ty = np.array([r.tilt_y_deg for r in R])
        om = np.array([r.omega for r in R]); ka = np.array([r.kappa for r in R])
        s["tilt_deg_median"] = float(np.median([r.tilt_deg for r in R]))
        s["corr_tilt_x_vs_omega"] = float(np.corrcoef(tx, om)[0, 1])
        s["corr_tilt_y_vs_kappa"] = float(np.corrcoef(ty, ka)[0, 1])
        s["corr_tilt_x_vs_kappa"] = float(np.corrcoef(tx, ka)[0, 1])
        s["corr_tilt_y_vs_omega"] = float(np.corrcoef(ty, om)[0, 1])
        yc = np.array([r.yaw_change_deg for r in R]); dp = np.array([r.dphi1 for r in R])
        s["corr_yaw_change_vs_dphi1"] = float(np.corrcoef(yc, dp)[0, 1]) if np.std(dp) > 0 else None
        zero = np.zeros(2)
        s["pos_error_center_m_median"] = float(np.median(_pos_errors(R, zero, "resid_center_px")))
        s["pos_error_nadir_m_median"] = float(np.median(_pos_errors(R, zero, "resid_nadir_px")))
        # Constant camera offset calibrated on the OTHER flight(s).
        others = [r for g in flights if g != f for r in ok_rows[g]]
        if others:
            bias = np.median(np.array([r.resid_nadir_px for r in others]), axis=0)
            s["calib_bias_px_from_other"] = bias.tolist()
            s["pos_error_nadir_calibrated_m_median"] = float(np.median(_pos_errors(R, bias, "resid_nadir_px")))
            e = _pos_errors(R, bias, "resid_nadir_px")
            s["pos_error_nadir_calibrated_m_p90"] = float(np.percentile(e, 90))
            c = _pos_errors(R, zero, "resid_center_px")
            s["pos_error_center_m_p90"] = float(np.percentile(c, 90))
        own = np.median(np.array([r.resid_nadir_px for r in R]), axis=0)
        s["own_bias_px"] = own.tolist()
        out[f] = s
    return out


def format_summary(summary: dict) -> str:
    lines = []
    for f, s in summary.items():
        if not s.get("ok"):
            lines.append(f"flight {f}: no usable pairs")
            continue
        lines += [
            f"flight {f}: {s['ok']}/{s['pairs']} pairs estimated",
            f"  speed error median {s['speed_error_pct_median']:.1f}% (bias {s['speed_bias_pct']:+.1f}%), "
            f"travel bearing error median {s['bearing_error_deg_median']:.1f} deg",
            f"  tilt median {s['tilt_deg_median']:.2f} deg; corr tilt_x~Omega {s['corr_tilt_x_vs_omega']:+.2f}, "
            f"tilt_y~Kappa {s['corr_tilt_y_vs_kappa']:+.2f} (cross: x~Kappa {s['corr_tilt_x_vs_kappa']:+.2f}, "
            f"y~Omega {s['corr_tilt_y_vs_omega']:+.2f}); yaw change~dPhi1 {s['corr_yaw_change_vs_dphi1']}",
            f"  position vs CSV, median: image center {s['pos_error_center_m_median']:.1f} m, "
            f"nadir pixel {s['pos_error_nadir_m_median']:.1f} m"
            + (f", nadir + offset calibrated on other flight {s['pos_error_nadir_calibrated_m_median']:.1f} m "
               f"(p90 {s['pos_error_nadir_calibrated_m_p90']:.1f} vs center {s['pos_error_center_m_p90']:.1f})"
               if "pos_error_nadir_calibrated_m_median" in s else ""),
            f"  constant offset (px): this flight {np.round(s['own_bias_px'], 0).tolist()}"
            + (f", other {np.round(s['calib_bias_px_from_other'], 0).tolist()}" if "calib_bias_px_from_other" in s else ""),
        ]
    return "\n".join(lines)


def run(flights: list[Flight], step: int, out_dir: Path, cache_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {}
    for fl in flights:
        print(f"flight {fl.flight_id}", flush=True)
        rows[fl.flight_id] = evaluate_flight(fl, step, cache_dir)
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "pairs.json").write_text(json.dumps({f: [asdict(r) for r in rs] for f, rs in rows.items()}))
    return summary
