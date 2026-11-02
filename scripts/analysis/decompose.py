"""Run the error-budget decomposition. Usage: decompose.py [--vocheck runs/<dir>] FLIGHT..."""
import argparse
from pathlib import Path

from gpsloc import error_budget as eb
from gpsloc.dataset import Flight

ap = argparse.ArgumentParser()
ap.add_argument("flights", nargs="+")
ap.add_argument("--vocheck")
ap.add_argument("--variant", action="append")
a = ap.parse_args()
root = Path.home() / "data/UAV-VisLoc"
G = {f: eb.geometry(Flight(root, f), Path("runs/cache")) for f in a.flights}
for kind in ("center", "csv", "mappose"):
    rows = {f: [(g, g[kind]) for g in G[f].values()] for f in a.flights}
    print(eb.format_result(eb.analyze(rows), kind))
if a.vocheck:
    for v in a.variant or ["sift1600_both"]:
        rows = eb.rows_from_vocheck(Path(a.vocheck) / "frames.json", G, v)
        print(eb.format_result(eb.analyze(rows), "vision " + v))
