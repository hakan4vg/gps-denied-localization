"""Height above ground from terrain relief: camera parallax (relative) x DEM (meters).

After the ground-plane homography H between frames k and k+1, a point that sits h meters above
(or below) that plane is left with a residual

    p2 - H p1  ~  f * (t/d)_xy * h / d          (near-nadir view, h << d)

along the direction of travel. t/d is known from the decomposition, so every tracked point
gives rho = h / d: its height above the plane in units of the flight height, no ML and no
labeling of features needed.

The DEM gives the same points' heights above the plane in meters. Regressing DEM relief on rho
gives d, the camera's distance to the ground: absolute altitude without an altimeter.

Needs: a rough position (to look up the DEM; tracking mode), real relief in the footprint
(tens of meters), and a DEM of the area (a 10 x 10 km tile at 30 m is ~220 KB as int16).
Compute: one residual and one dot product per tracked point, plus a 1-parameter fit.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .dataset import Flight
from .dem import elevation, height_above_ground
from .motion import intrinsics, to_gray
from .repeatability import aligned_frames


def tracks(g1, g2, n_points=400, grid=(10, 8)):
    """Grid-spread corners tracked with pyramidal LK, forward-backward checked (same as motion)."""
    h, w = g1.shape
    gx, gy = grid
    per = max(1, n_points // (gx * gy))
    pts = []
    for j in range(gy):
        for i in range(gx):
            x0, x1, y0, y1 = i * w // gx, (i + 1) * w // gx, j * h // gy, (j + 1) * h // gy
            c = cv2.goodFeaturesToTrack(g1[y0:y1, x0:x1], per, 0.01, max(3, w // 80), blockSize=5)
            if c is not None:
                pts.append(c.reshape(-1, 2) + (x0, y0))
    p1 = np.concatenate(pts).astype(np.float32)
    from .motion import lk_track

    p2, _, ok = lk_track(g1, g2, p1)
    return p1[ok].astype(np.float64), p2[ok].astype(np.float64)


def relative_heights(p1, p2, K):
    """rho = h/d per point, plus the homography fit. p1/p2 in full-res pixels."""
    if len(p1) < 30:
        return None
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0, maxIters=500)
    if H is None or mask.sum() < 30:
        return None
    from .motion import _decompose

    inl = mask.ravel().astype(bool)
    m = _decompose(H / H[2, 2], K, p1[inl])
    if not m.ok:
        return None
    proj = cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = p2 - proj
    # Residual direction for a raised point: along f * (t/d)_xy in image k+1 axes.
    tdir = K[0, 0] * m.t_over_d[:2]
    rho = resid @ tdir / (tdir @ tdir)  # h/d, positive = raised toward the camera
    # Perpendicular residual is noise (or moving objects): use it to flag bad points.
    perp = resid - np.outer(rho, tdir)
    return rho, np.linalg.norm(perp, axis=1), m


def altitude_for_frame(fk, f2, al, sat, width=640):
    """Estimate d from relief; returns (d_est, agl_truth, n_points, dem_relief_sd)."""
    g1, s = to_gray(fk.load_image(), width)
    g2, _ = to_gray(f2.load_image(), width)
    K = intrinsics(round(g1.shape[1] / s), round(g1.shape[0] / s))
    p1, p2 = tracks(g1, g2)
    p1, p2 = p1 / s, p2 / s
    out = relative_heights(p1, p2, K)
    if out is None:
        return None
    rho, perp, m = out
    keep = perp < 2.0 / s  # consistent with pure ground-relief parallax
    p1, rho = p1[keep], rho[keep]
    if len(rho) < 30:
        return None
    # DEM elevation under each tracked point (position from the reference alignment).
    mp = al.drone_to_map(p1)
    ll = [sat.pixel_to_latlon(r, c) for c, r in mp]
    z = np.array([elevation(lat, lon) for lat, lon in ll])
    # Heights above the best-fit plane, in meters (DEM) and in units of d (camera).
    A = np.column_stack([p1, np.ones(len(p1))])
    z_res = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    r_res = rho - A @ np.linalg.lstsq(A, rho, rcond=None)[0]
    if np.std(z_res) < 3.0:
        return {"skip": "flat footprint", "dem_relief_sd": float(np.std(z_res))}
    d = float((r_res @ z_res) / (r_res @ r_res))  # z_res ~ d * r_res
    corr = float(np.corrcoef(r_res, z_res)[0, 1])
    return {"d_est": d, "agl": height_above_ground(fk), "n": int(len(rho)), "dem_relief_sd": float(np.std(z_res)),
            "corr": corr, "baseline_ratio": m.baseline_ratio}


def run(flights: list[Flight], step: int, out_dir: Path, cache_dir: Path, width: int = 640) -> dict:
    cv2.setNumThreads(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fl in flights:
        frames = list(fl.frames())
        idx = {f.image_path.name: i for i, f in enumerate(frames)}
        sat = fl.satellite_map()
        rows = []
        for fk, al in aligned_frames(fl, step, cache_dir, progress=False):
            i = idx[fk.image_path.name]
            if i + 1 >= len(frames):
                continue
            f2 = frames[i + 1]
            dt = (datetime.fromisoformat(f2.date) - datetime.fromisoformat(fk.date)).total_seconds()
            if not (0 < dt <= 15):
                continue
            r = altitude_for_frame(fk, f2, al, sat, width)
            if r:
                r["frame"] = fk.image_path.name
                rows.append(r)
                if "d_est" in r:
                    print(f"  {fl.flight_id} {r['frame']}: d_est {r['d_est']:.0f} m vs {r['agl']:.0f} m "
                          f"(relief sd {r['dem_relief_sd']:.1f} m, corr {r['corr']:.2f}, n {r['n']})", flush=True)
        est = [r for r in rows if "d_est" in r]
        s = {"frames": len(rows), "estimated": len(est)}
        if est:
            e = np.array([r["d_est"] / r["agl"] - 1 for r in est])
            good = [r for r in est if r["corr"] > 0.5]
            s.update({"error_pct_median_abs": float(100 * np.median(np.abs(e))), "bias_pct": float(100 * np.median(e)),
                      "relief_sd_median": float(np.median([r["dem_relief_sd"] for r in est]))})
            if good:
                eg = np.array([r["d_est"] / r["agl"] - 1 for r in good])
                s.update({"confident": len(good), "confident_error_pct_median_abs": float(100 * np.median(np.abs(eg)))})
        summary[fl.flight_id] = s
        print(f"flight {fl.flight_id}: {json.dumps(s)}", flush=True)
        (out_dir / f"rows_{fl.flight_id}.json").write_text(json.dumps(rows))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
