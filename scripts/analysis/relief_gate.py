"""Is vision-nadir error explained by non-flat ground in the footprint? DEM-based gate.

Non-flatness = sd of DEM heights around the best-fit plane within the footprint, divided by
height above ground (the parallax the homography can't model).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from gpsloc import dem
from gpsloc.dataset import Flight

root = Path.home() / "data/UAV-VisLoc"
vdir = Path(sys.argv[1])
VIS = sys.argv[2]
d = json.loads((vdir / "frames.json").read_text())


def flatness(lat, lon, radius_m):
    src = dem._src(lat, lon, dem.DEFAULT_DIR)
    col, row = ~src.transform * (lon, lat)
    py = abs(src.transform.e) * 111_320.0
    px = abs(src.transform.a) * 111_320.0 * math.cos(math.radians(lat))
    rx, ry = int(math.ceil(radius_m / px)), int(math.ceil(radius_m / py))
    a = src.read(1, window=Window(int(col) - rx, int(row) - ry, 2 * rx + 1, 2 * ry + 1), boundless=True,
                 fill_value=0).astype(float)
    yy, xx = np.mgrid[-ry:ry + 1, -rx:rx + 1]
    e, n = xx * px, -yy * py
    m = e**2 + n**2 <= radius_m**2
    A = np.column_stack([e[m], n[m], np.ones(m.sum())])
    c, *_ = np.linalg.lstsq(A, a[m], rcond=None)
    return float(np.std(a[m] - A @ c))


rows = []
for f, rs in d.items():
    if not rs:
        continue
    fr = {x.image_path.name: x for x in Flight(root, f).frames()}
    others = [r for g, o in d.items() if g != f for r in o if r["variants"].get(VIS, {}).get("ok")]
    b = np.median([r["variants"][VIS]["resid_px"] for r in others], axis=0)
    for r in rs:
        v = r["variants"].get(VIS, {})
        if not v.get("ok") or not r["agl"]:
            continue
        x = fr[r["frame"]]
        sd = flatness(x.truth.lat, x.truth.lon, radius_m=0.6 * r["agl"])  # ~ half footprint width
        err = float(np.hypot(*(np.array(v["resid_px"]) - b)) * r["gsd_m"])
        rows.append((f, sd / r["agl"], err))
rows = np.array(rows, dtype=object)
ratio = rows[:, 1].astype(float)
err = rows[:, 2].astype(float)
for f in dict.fromkeys(rows[:, 0]):
    m = rows[:, 0] == f
    print(f"flight {f}: relief sd/AGL median {np.median(ratio[m]) * 100:.2f}%, error median {np.median(err[m]):.1f} m")
rho = np.corrcoef(np.argsort(np.argsort(ratio)), np.argsort(np.argsort(err)))[0, 1]
print(f"spearman(error, relief/AGL) over all frames: {rho:+.2f}")
print("gate: keep frames with relief sd/AGL below threshold")
for th in (0.5, 1, 1.5, 2, 3, 5, 100):
    k = ratio * 100 < th
    print(f"   < {th:5.1f}%: availability {k.mean():5.0%}, median {np.median(err[k]):5.1f} m, "
          f"p90 {np.percentile(err[k], 90):5.1f} m, >30 m wrong {np.mean(err[k] > 30):5.1%}"
          + "  by flight: " + ", ".join(f"{f} {np.mean(k[rows[:, 0] == f]):.0%}" for f in dict.fromkeys(rows[:, 0])))
