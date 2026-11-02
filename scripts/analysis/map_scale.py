"""Map-pose sanity: height vs DEM height above ground, tilt vs CSV."""
import sys
from pathlib import Path
import numpy as np
from gpsloc import error_budget as eb
from gpsloc.dataset import Flight
from gpsloc.dem import usable_agl
root = Path.home() / "data/UAV-VisLoc"
for f in sys.argv[1:]:
    fl = Flight(root, f)
    G = eb.geometry(fl, Path("runs/cache"))
    fr = {x.image_path.name: x for x in fl.frames()}
    h = np.array([(g["mappose_height"], usable_agl(fr[n]) or np.nan) for n, g in G.items()])
    t = np.array([g["mappose_tilt"] + [fr[n].sensors["Omega"], fr[n].sensors["Kappa"]] for n, g in G.items()])
    ratio = h[:, 0] / h[:, 1]
    print(f"flight {f}: map-pose height / DEM AGL median {np.nanmedian(ratio):.3f}, sd {np.nanstd(ratio):.3f}")
    for k, nm in ((0, "x~Omega"), (1, "y~Kappa")):
        for sgn in (1, -1):
            d = t[:, k] - sgn * t[:, 2 + k]
            print(f"   tilt {nm} sign {sgn:+d}: corr {np.corrcoef(t[:, k], sgn * t[:, 2 + k])[0, 1]:+.2f}, "
                  f"offset {np.median(d):+.2f}, sd {np.std(d - np.median(d)):.2f} deg")
