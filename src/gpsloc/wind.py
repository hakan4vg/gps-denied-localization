"""Airspeed and wind from camera motion alone (wind triangle).

A fixed-wing drone flies at a roughly constant airspeed along its nose; the wind adds a constant
vector. Ground velocity = air velocity + wind. On a back-and-forth survey the nose points in
different directions, so ground speed swings (10-32 m/s on these flights) and both unknowns can
be solved.

Camera-only version:
- each frame pair gives the ground velocity in the camera's axes (direction exact; magnitude in
  heights per second, x height for m/s) and the heading change between frames;
- integrating heading changes expresses every velocity in the first frame's axes (heading drift
  accumulates slowly);
- with the camera fixed to the airframe, the air velocity is a constant vector b in camera axes,
  so  v_j(frame 0) = Rot(-Gamma_j) b + w  is linear in (b, w): 4 unknowns, least squares.

Reference version (validation): same model with GPS ground velocity and compass heading (CSV
Phi1) in north/east axes.

Compute: one 4x4 normal-equation solve per update; trivial for a microcontroller.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .dataset import Flight
from .dem import usable_agl
from .geo import haversine_m
from .motion import intrinsics, to_gray, track_pair


def _rot(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


@dataclass
class PairVel:
    frame: str
    ok: bool
    dt: float
    yaw_change_deg: float | None = None
    v_cam: tuple[float, float] | None = None  # m/s in camera k axes (x right, y down)
    v_gps_ne: tuple[float, float] | None = None  # m/s (north, east)
    heading_csv: float | None = None
    agl: float | None = None


def pair_velocities(flight: Flight, width: int = 480, max_frames: int | None = None,
                    progress: bool = True) -> list[PairVel]:
    frames = list(flight.frames())[: max_frames]
    out: list[PairVel] = []
    prev = None
    K = None
    for k in range(len(frames) - 1):
        f1, f2 = frames[k], frames[k + 1]
        dt = (datetime.fromisoformat(f2.date) - datetime.fromisoformat(f1.date)).total_seconds()
        north = haversine_m(f1.truth.lat, f1.truth.lon, f2.truth.lat, f1.truth.lon) * np.sign(f2.truth.lat - f1.truth.lat)
        east = haversine_m(f1.truth.lat, f1.truth.lon, f1.truth.lat, f2.truth.lon) * np.sign(f2.truth.lon - f1.truth.lon)
        pv = PairVel(f1.image_path.name, False, dt, heading_csv=f1.sensors.get("Phi1"),
                     v_gps_ne=(north / dt, east / dt) if dt > 0 else None)
        if not (0 < dt <= 15):
            out.append(pv)
            prev = None
            continue
        g1, s = prev if prev is not None else to_gray(f1.load_image(), width)
        g2, _ = to_gray(f2.load_image(), width)
        prev = (g2, s)
        if K is None:
            K = intrinsics(round(g1.shape[1] / s), round(g1.shape[0] / s))
        m = track_pair(g1, g2, s, K)
        agl = usable_agl(f1)
        if m.ok and agl is not None:
            # Height only converts heights/s to m/s. Here from CSV altitude - DEM (analysis);
            # onboard it would come from the map-match scale.
            mv = m.motion_cam_k[:2] * agl / dt
            pv.ok, pv.yaw_change_deg, pv.v_cam, pv.agl = True, m.yaw_change_deg, (float(mv[0]), float(mv[1])), agl
        out.append(pv)
        if progress and k % 50 == 0:
            print(f"  {flight.flight_id} {k}/{len(frames) - 1}", flush=True)
    return out


def solve_wind(vels: list[np.ndarray], rots: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares for v_j = rots_j @ b + w. Returns b (air velocity, body axes), w, residuals."""
    A = np.zeros((2 * len(vels), 4))
    y = np.zeros(2 * len(vels))
    for j, (v, R) in enumerate(zip(vels, rots)):
        A[2 * j:2 * j + 2, :2] = R
        A[2 * j:2 * j + 2, 2:] = np.eye(2)
        y[2 * j:2 * j + 2] = v
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = (y - A @ x).reshape(-1, 2)
    return x[:2], x[2:], res


def analyze(pairs: list[PairVel], max_turn_deg: float = 8.0) -> dict:
    # Camera-only: integrate heading changes over ALL tracked pairs (turns included), but fit
    # only straight segments.
    vels_c, rots_c, gamma, idx_c = [], [], 0.0, []
    for j, p in enumerate(pairs):
        if not p.ok:
            continue  # a gap breaks nothing here except the heading chain, which we keep
        if abs(p.yaw_change_deg) <= max_turn_deg:
            # velocity measured in camera-k axes -> frame-0 axes
            Rk = _rot(-gamma)
            vels_c.append(Rk @ np.array(p.v_cam))
            rots_c.append(Rk)
            idx_c.append(j)
        gamma += p.yaw_change_deg
    b_c, w_c, res_c = solve_wind(vels_c, rots_c)

    # Reference: GPS velocity (north, east) and compass heading Phi1 (clockwise from north).
    vels_g, rots_g = [], []
    for j in idx_c:
        p = pairs[j]
        vels_g.append(np.array(p.v_gps_ne))
        rots_g.append(_rot(p.heading_csv))
    b_g, w_g, res_g = solve_wind(vels_g, rots_g)

    # Per-pair ground speed predicted by each model vs GPS ground speed.
    gs_gps = np.array([np.hypot(*pairs[j].v_gps_ne) for j in idx_c])
    gs_cam = np.array([np.linalg.norm(R @ b_c + w_c) for R in rots_c])
    gs_cam_meas = np.array([np.linalg.norm(v) for v in vels_c])
    return {
        "pairs_used": len(idx_c),
        "heading_drift_deg_total": float(gamma),
        "camera": {"airspeed": float(np.linalg.norm(b_c)), "wind_speed": float(np.linalg.norm(w_c)),
                   "crab_axis_deg": float(math.degrees(math.atan2(b_c[1], b_c[0]))),
                   "fit_rms_m_s": float(np.sqrt(np.mean(np.sum(res_c**2, 1))))},
        "gps_reference": {"airspeed": float(np.linalg.norm(b_g)), "wind_speed": float(np.linalg.norm(w_g)),
                          "wind_from_deg": float((math.degrees(math.atan2(-w_g[1], -w_g[0]))) % 360),
                          "fit_rms_m_s": float(np.sqrt(np.mean(np.sum(res_g**2, 1))))},
        "ground_speed_gps_range": [float(gs_gps.min()), float(gs_gps.max())] if len(gs_gps) else None,
        "ground_speed_camera_error_median_pct": float(100 * np.median(np.abs(gs_cam_meas / gs_gps - 1))),
        "ground_speed_model_error_median_pct": float(100 * np.median(np.abs(gs_cam / gs_gps - 1))),
    }


def format_result(fid: str, r: dict) -> str:
    c, g = r["camera"], r["gps_reference"]
    return (
        f"flight {fid}: {r['pairs_used']} straight pairs, ground speed {r['ground_speed_gps_range'][0]:.1f}-"
        f"{r['ground_speed_gps_range'][1]:.1f} m/s (GPS)\n"
        f"  camera only : airspeed {c['airspeed']:.1f} m/s, wind {c['wind_speed']:.1f} m/s, fit rms {c['fit_rms_m_s']:.1f} m/s "
        f"(heading chain drift-free check: total turn {r['heading_drift_deg_total']:.0f} deg)\n"
        f"  GPS+compass : airspeed {g['airspeed']:.1f} m/s, wind {g['wind_speed']:.1f} m/s from {g['wind_from_deg']:.0f} deg, "
        f"fit rms {g['fit_rms_m_s']:.1f} m/s\n"
        f"  ground speed error vs GPS: measured {r['ground_speed_camera_error_median_pct']:.1f}%, "
        f"wind-model {r['ground_speed_model_error_median_pct']:.1f}% (median)"
    )


def run(flights: list[Flight], out_dir: Path, width: int = 480) -> dict:
    cv2.setNumThreads(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for fl in flights:
        pairs = pair_velocities(fl, width)
        r = analyze(pairs)
        results[fl.flight_id] = r
        print(format_result(fl.flight_id, r), flush=True)
        (out_dir / f"pairs_{fl.flight_id}.json").write_text(json.dumps([asdict(p) for p in pairs]))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    return results
