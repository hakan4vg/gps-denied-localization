"""Split the remaining position error into terms that behave differently as the drone turns
(analysis only).

Residual r = (CSV position seen in the drone image) - (predicted drone position), in meters
along the image axes. Model per flight:

    r = b + R(heading) m + L v + noise

- b: constant in image PIXELS (an angle, so meters grow with height) -> camera mount angle or
  principal point (hardware constant);
- m: constant in east/north -> satellite map georeferencing (or GPS) offset (per map);
- L: seconds, times the ground velocity v in image axes -> lag between photo and GPS fix.

The terms separate only because heading varies within a flight (out-and-back legs).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .dataset import Flight
from .motion import intrinsics
from .vo_validation import _nadir_csv

IMG_W, IMG_H = 3976, 2652  # every UAV-VisLoc frame
MODELS = {"b": ("b",), "b+m": ("b", "m"), "b+L": ("b", "L"), "b+m+L": ("b", "m", "L")}


def geometry(flight: Flight, cache_dir: Path) -> dict[str, dict]:
    """Per aligned frame: residuals (center, CSV-tilt nadir) and the regressors, in meters."""
    data = json.loads((Path(cache_dir) / f"align_{flight.flight_id}.json").read_text())
    frames = list(flight.frames())
    idx = {f.image_path.name: i for i, f in enumerate(frames)}
    sat = flight.satellite_map()
    gsd_map = sat.gsd_m()
    K = intrinsics(IMG_W, IMG_H)
    out = {}
    for name, a in data.items():
        if not a["ok"]:
            continue
        i = idx[name]
        if i == 0 or i + 1 >= len(frames):
            continue
        f0, fm, fp = frames[i], frames[i - 1], frames[i + 1]
        t = [datetime.fromisoformat(f.date) for f in (fm, f0, fp)]
        dt = (t[2] - t[0]).total_seconds()
        if not 0 < dt <= 30:
            continue
        Hinv = np.linalg.inv(np.array(a["H"]))

        def to_img(row, col):
            p = Hinv @ np.array([col, row, 1.0])
            return p[:2] / p[2]

        r, c = sat.latlon_to_pixel(f0.truth.lat, f0.truth.lon)
        truth = to_img(r, c)
        gsd = a["scale"] * gsd_map  # meters per drone pixel
        # 1 m east and 1 m north on the ground, as image vectors in meters along image axes.
        e_img = (to_img(r, c + 1 / gsd_map) - truth) * gsd
        n_img = (to_img(r - 1 / gsd_map, c) - truth) * gsd
        vm = to_img(*sat.latlon_to_pixel(fm.truth.lat, fm.truth.lon))
        vp = to_img(*sat.latlon_to_pixel(fp.truth.lat, fp.truth.lon))
        s = f0.sensors
        pose = pose_from_map_homography(np.array(a["H"]), K, (r, c), gsd_map)
        out[name] = {
            "gsd": gsd,
            "R": np.column_stack([e_img, n_img]),
            "v": (vp - vm) * gsd / dt,
            "center": (truth - np.array([IMG_W / 2, IMG_H / 2])) * gsd,
            "csv": (truth - _nadir_csv(s["Omega"], s["Kappa"], K)) * gsd,
            "mappose": (truth - pose["nadir_px"]) * gsd,
            "mappose_height": pose["height"],
            "mappose_tilt": pose["tilt_xy_deg"],
            "truth_px": truth,
        }
    return out


def pose_from_map_homography(H: np.ndarray, K: np.ndarray, origin_rc, gsd_map: float) -> dict:
    """Camera pose from the drone->orthophoto homography alone (planar PnP).

    Ground frame: X east, Y north (meters) around map pixel `origin_rc`, Z up, ground at Z=0.
    ground(X, Y, 1) ~ A @ drone_px with A = S @ H; so drone_px ~ K [r1 r2 t] ground.
    """
    r0, c0 = origin_rc
    S = np.array([[gsd_map, 0, -gsd_map * c0], [0, -gsd_map, gsd_map * r0], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ np.linalg.inv(S @ H)
    M /= np.sqrt(np.linalg.norm(M[:, 0]) * np.linalg.norm(M[:, 1]))
    if M[2, 2] < 0:  # ground must be in front of the camera
        M = -M
    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt  # world -> camera
    C = -R.T @ t  # camera center in the ground frame
    below = np.array([C[0], C[1], 0.0])  # plumb point (flat-ground assumption)
    p = K @ (R @ below + t)
    nadir = p[:2] / p[2]
    down_cam = R @ np.array([0, 0, -1.0])  # true vertical in camera axes
    tilt = np.degrees([np.arctan2(down_cam[0], down_cam[2]), np.arctan2(down_cam[1], down_cam[2])])
    return {"nadir_px": nadir, "height": float(abs(C[2])), "tilt_xy_deg": tilt.tolist()}


def design(g: dict, terms: tuple[str, ...]) -> np.ndarray:
    cols = []
    if "b" in terms:
        cols.append(g["gsd"] * np.eye(2))  # b in pixels
    if "m" in terms:
        cols.append(g["R"])
    if "L" in terms:
        cols.append(g["v"][:, None])
    return np.hstack(cols)


def fit(rows: list[tuple[dict, np.ndarray]], terms) -> np.ndarray:
    """Least squares, reweighted so a few bad registrations don't dominate."""
    A = np.vstack([design(g, terms) for g, _ in rows])
    y = np.concatenate([r for _, r in rows])
    w = np.ones(len(y))
    for _ in range(3):
        x, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
        e = (y - A @ x).reshape(-1, 2)
        d = np.repeat(np.hypot(e[:, 0], e[:, 1]), 2)
        w = 1 / np.sqrt(np.maximum(d, np.median(d)))
    return x


def unpack(x, terms) -> dict:
    out, k = {}, 0
    for t in terms:
        if t == "L":
            out[t] = float(x[k])
            k += 1
        else:
            out[t] = [float(x[k]), float(x[k + 1])]
            k += 2
    return out


def apply(rows, terms, x) -> np.ndarray:
    return np.array([np.hypot(*(r - design(g, terms) @ x)) for g, r in rows])


def loo_errors(rows, terms) -> np.ndarray:
    """Leave-one-frame-out: fit on the rest of the flight, score the held-out frame."""
    e = []
    for j, (g, r) in enumerate(rows):
        x = fit(rows[:j] + rows[j + 1:], terms)
        e.append(np.hypot(*(r - design(g, terms) @ x)))
    return np.array(e)


def analyze(rows_by_flight: dict[str, list[tuple[dict, np.ndarray]]]) -> dict:
    res = {}
    for f, rows in rows_by_flight.items():
        if len(rows) < 12:
            continue
        r = {"n": len(rows), "raw_median": float(np.median([np.hypot(*y) for _, y in rows])),
             "speed_median": float(np.median([np.hypot(*g["v"]) for g, _ in rows])), "models": {}}
        # Direction of travel in image axes: if it never changes, lag and mount offset can only
        # be told apart through speed changes.
        ang = np.degrees([np.arctan2(g["v"][1], g["v"][0]) for g, _ in rows])
        r["travel_dir_img_deg"] = [float(np.median(ang)), float(np.std(ang))]
        r["heading_spread_deg"] = float(np.std(np.degrees([np.arctan2(*g["R"][:, 1]) for g, _ in rows])))
        for name, terms in MODELS.items():
            x = fit(rows, terms)
            e = loo_errors(rows, terms)
            r["models"][name] = {"params": unpack(x, terms), "loo_median": float(np.median(e)),
                                 "loo_p90": float(np.percentile(e, 90))}
        # Hardware constant b (mount angle) from the OTHER flights; map offset fitted per flight.
        # (Lag is left out: the camera is body-fixed, so travel direction in image axes barely
        # changes and lag is nearly collinear with b.)
        others = [row for g2, rs in rows_by_flight.items() if g2 != f for row in rs]
        if len(others) >= 12:
            xo = unpack(fit(others, ("b",)), ("b",))
            corr = [(g, y - g["gsd"] * np.array(xo["b"])) for g, y in rows]
            e_hw = np.array([np.hypot(*y) for _, y in corr])
            e_hw_m = loo_errors(corr, ("m",))
            r["cross_flight"] = {"b": xo["b"],
                                 "median": float(np.median(e_hw)), "p90": float(np.percentile(e_hw, 90)),
                                 "plus_map_median": float(np.median(e_hw_m)),
                                 "plus_map_p90": float(np.percentile(e_hw_m, 90))}
        res[f] = r
    return res


def format_result(res: dict, label: str) -> str:
    lines = [f"== {label}  (leave-one-frame-out median / p90, m)"]
    for f, r in res.items():
        lines.append(f"flight {f}: n={r['n']}, raw median {r['raw_median']:.1f} m, "
                     f"speed {r['speed_median']:.0f} m/s, travel dir in image "
                     f"{r['travel_dir_img_deg'][0]:.0f}+-{r['travel_dir_img_deg'][1]:.0f} deg, "
                     f"map heading sd {r['heading_spread_deg']:.0f} deg")
        for name, m in r["models"].items():
            p = m["params"]
            desc = []
            if "b" in p:
                desc.append(f"b=({p['b'][0]:+.0f},{p['b'][1]:+.0f}) px")
            if "m" in p:
                desc.append(f"map E{p['m'][0]:+.1f} N{p['m'][1]:+.1f} m")
            if "L" in p:
                desc.append(f"lag {p['L']:+.2f} s")
            lines.append(f"   {name:6} {m['loo_median']:5.1f} / {m['loo_p90']:5.1f}   " + ", ".join(desc))
        if "cross_flight" in r:
            c = r["cross_flight"]
            lines.append(f"   b from other flights (({c['b'][0]:+.0f},{c['b'][1]:+.0f}) px): "
                         f"{c['median']:.1f} / {c['p90']:.1f};  + per-map offset: "
                         f"{c['plus_map_median']:.1f} / {c['plus_map_p90']:.1f}")
    return "\n".join(lines)


def rows_from_vocheck(frames_json: Path, G: dict[str, dict], variant: str) -> dict[str, list]:
    """Pair vocheck's per-frame vision residuals (pixels) with this module's geometry."""
    d = json.loads(Path(frames_json).read_text())
    out = {}
    for f, rs in d.items():
        rows = []
        for r in rs:
            g = G.get(f, {}).get(r["frame"])
            v = r["variants"].get(variant, {})
            if g is not None and v.get("ok"):
                rows.append((g, np.array(v["resid_px"]) * g["gsd"]))
        out[f] = rows
    return out
