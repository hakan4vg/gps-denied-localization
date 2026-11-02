"""What do the worst frames have in common? Residual after mount angle + map offset vs candidates."""
import json
import sys
from pathlib import Path

import numpy as np

from gpsloc import error_budget as eb
from gpsloc.dataset import Flight

root = Path.home() / "data/UAV-VisLoc"
for f in sys.argv[1:]:
    fl = Flight(root, f)
    G = eb.geometry(fl, Path("runs/cache"))
    cache = json.loads(Path(f"runs/cache/align_{f}.json").read_text())
    frames = list(fl.frames())
    idx = {fr.image_path.name: i for i, fr in enumerate(frames)}
    rows = [(g, g["csv"]) for g in G.values()]
    names = list(G)
    x = eb.fit(rows, ("b", "m"))
    feats = []
    for n, (g, y) in zip(names, rows):
        e = y - eb.design(g, ("b", "m")) @ x
        i = idx[n]
        s, sp, sm = frames[i].sensors, frames[i + 1].sensors, frames[i - 1].sensors
        dphi = ((sp["Phi1"] - sm["Phi1"] + 180) % 360) - 180
        feats.append({
            "frame": n, "err": float(np.hypot(*e)), "ex": float(e[0]), "ey": float(e[1]),
            "tilt": float(np.hypot(s["Omega"], s["Kappa"])), "omega": s["Omega"], "kappa": s["Kappa"],
            "turn": abs(dphi), "d_omega": abs(sp["Omega"] - sm["Omega"]), "d_kappa": abs(sp["Kappa"] - sm["Kappa"]),
            "inliers": cache[n]["inliers"], "speed": float(np.hypot(*g["v"])), "gsd": g["gsd"],
        })
    err = np.array([q["err"] for q in feats])
    print(f"flight {f}: median {np.median(err):.1f} m, p90 {np.percentile(err, 90):.1f} m")
    for k in ("tilt", "turn", "d_omega", "d_kappa", "inliers", "speed", "gsd"):
        v = np.array([q[k] for q in feats], float)
        rho = np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(err)))[0, 1]
        print(f"   spearman(err, {k:8}) = {rho:+.2f}")
    print("   worst frames:")
    for q in sorted(feats, key=lambda q: -q["err"])[:8]:
        print(f"     {q['frame']} err {q['err']:5.1f} (x{q['ex']:+5.1f} y{q['ey']:+5.1f})  omega {q['omega']:+5.1f} "
              f"kappa {q['kappa']:+5.1f}  turn {q['turn']:5.1f}  dOm {q['d_omega']:4.1f} dKa {q['d_kappa']:4.1f}  "
              f"inl {q['inliers']}")
