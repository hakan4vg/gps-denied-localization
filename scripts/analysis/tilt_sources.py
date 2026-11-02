"""Tilt sources compared and combined, on vocheck output. Usage: tilt_sources.py runs/<vocheck dir>

1. Three-cornered hat: noise of CSV tilt and two vision tilts without ground truth.
2. Calm vs fast-changing attitude: where does each source fail?
3. Blend vision and CSV nadir (weight chosen on the other flights), scored with the
   error-budget models (mount angle from other flights, + per-map offset).
"""
import json
import sys
from pathlib import Path

import numpy as np

from gpsloc import error_budget as eb
from gpsloc.dataset import Flight

root = Path.home() / "data/UAV-VisLoc"
vdir = Path(sys.argv[1])
VIS = sys.argv[2] if len(sys.argv) > 2 else "sift1600_both"
d = json.loads((vdir / "frames.json").read_text())
flights = list(d)


def err_m(px, gsd):
    return np.hypot(*px) * gsd


# --- 1. three-cornered hat on tilt (degrees) ------------------------------------------------
print("== three-cornered hat: own noise sd of each tilt source (deg), offsets removed")
A_, B_ = "sift1600", "sift1600_k+2"  # pairs (k,k+1) and (k,k+2): different partner frames
for f in flights:
    rows = [r for r in d[f] if r["variants"].get(A_, {}).get("ok") and r["variants"].get(B_, {}).get("ok")]
    if len(rows) < 10:
        continue
    out = []
    for ax, csvk in (("tilt_x", "omega"), ("tilt_y", "kappa")):
        c = np.array([r[csvk] for r in rows])
        a = np.array([r["variants"][A_][ax] for r in rows])
        b = np.array([r["variants"][B_][ax] for r in rows])
        v = lambda x, y: np.var((x - y) - np.mean(x - y))
        vca, vcb, vab = v(c, a), v(c, b), v(a, b)
        sc = (vca + vcb - vab) / 2
        sa = (vca + vab - vcb) / 2
        sb = (vcb + vab - vca) / 2
        f_ = lambda s: f"{np.sqrt(s):.2f}" if s > 0 else f"(neg {s:.2f})"
        out.append(f"{ax[-1]}: CSV {f_(sc)}, vis(k+1) {f_(sa)}, vis(k+2) {f_(sb)}")
    print(f"flight {f} (n={len(rows)}): " + " | ".join(out))
print("   (vision estimates share frame k, so their common error is counted as CSV noise:"
      " the CSV figure is an upper bound)")

# --- geometry for the error-budget models -----------------------------------------------------
G = {f: eb.geometry(Flight(root, f), Path("runs/cache")) for f in flights}


def bias_px(rows, key):
    pts = [r[key] for r in rows if r[key] is not None]
    return np.median(np.array(pts), axis=0) if pts else np.zeros(2)


def resid(r, kind):
    if kind == "csv":
        return np.array(r["resid_csv_px"])
    v = r["variants"].get(kind, {})
    return np.array(v["resid_px"]) if v.get("ok") else None


# --- 2. calm vs fast attitude change ------------------------------------------------------
print(f"\n== calm vs fast-changing attitude (|dOmega|+|dKappa| between k-1 and k+1), "
      f"mount offset from other flights, median m")
rate = {}
for f in flights:
    fr = list(Flight(root, f).frames())
    idx = {x.image_path.name: i for i, x in enumerate(fr)}
    for r in d[f]:
        i = idx[r["frame"]]
        a, b = fr[i - 1].sensors, fr[i + 1].sensors
        rate[(f, r["frame"])] = abs(b["Omega"] - a["Omega"]) + abs(b["Kappa"] - a["Kappa"])
for f in flights:
    others = [r for g in flights if g != f for r in d[g]]
    rows = [r for r in d[f] if resid(r, VIS) is not None]
    if len(rows) < 10:
        continue
    q = np.median([rate[(f, r["frame"])] for r in rows])
    line = f"flight {f} (split at {q:.1f} deg):"
    for kind in ("csv", VIS):
        bo = np.median([resid(r, kind) for r in others if resid(r, kind) is not None], axis=0)
        calm = [err_m(resid(r, kind) - bo, r["gsd_m"]) for r in rows if rate[(f, r["frame"])] <= q]
        fast = [err_m(resid(r, kind) - bo, r["gsd_m"]) for r in rows if rate[(f, r["frame"])] > q]
        line += f"  {kind}: calm {np.median(calm):.1f}, fast {np.median(fast):.1f}"
    print(line)

# --- 3. blend ----------------------------------------------------------------------------
print(f"\n== blend w*vision({VIS}) + (1-w)*CSV nadir; w chosen on other flights")
W = np.linspace(0, 1, 11)


def blended_rows(f, w):
    rows = []
    for r in d[f]:
        rv, rc = resid(r, VIS), resid(r, "csv")
        g = G[f].get(r["frame"])
        if rv is None or g is None:
            continue
        rows.append((g, (w * rv + (1 - w) * rc) * r["gsd_m"]))
    return rows


flights = [f for f in flights if len(blended_rows(f, 1.0)) >= 12]
for f in flights:
    others = [g for g in flights if g != f]
    if not others:
        continue
    # choose w on other flights: median error after their own mount-angle fit
    score = []
    for w in W:
        es = []
        for o in others:
            rows = blended_rows(o, w)
            if len(rows) >= 10:
                x = eb.fit(rows, ("b",))
                es += list(eb.apply(rows, ("b",), x))
        score.append(np.median(es) if es else np.inf)
    w_best = W[int(np.argmin(score))]
    res = {}
    for label, w in (("csv", 0.0), ("vision", 1.0), (f"blend w={w_best:.1f}", w_best)):
        rows_all = {g: blended_rows(g, w) for g in flights}
        r = eb.analyze(rows_all)[f]
        c = r["cross_flight"]
        res[label] = f"{c['median']:.1f}/{c['p90']:.1f} (+map {c['plus_map_median']:.1f}/{c['plus_map_p90']:.1f})"
    print(f"flight {f}: " + "   ".join(f"{k}: {v}" for k, v in res.items()))
print("   (median/p90 m; mount angle from other flights; '+map' adds a per-map offset, LOO)")
