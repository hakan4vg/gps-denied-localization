"""Command line entry point.

    gpsloc check --flight 03              # sanity-check data: GT inside map, map resolution, track plot
    gpsloc eval --flight 03 --method map-center [--step 10] [--limit 50]
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from . import baselines, evaluate
from .dataset import Flight, available_flights

DEFAULT_DATA = Path(os.environ.get("GPSLOC_DATA", Path.home() / "data" / "UAV-VisLoc"))


def cmd_check(args: argparse.Namespace) -> None:
    flight_ids = args.flight or available_flights(args.data)
    for fid in flight_ids:
        flight = Flight(args.data, fid)
        m = flight.satellite_map()
        frames = list(flight.frames())
        inside = 0
        for fr in frames:
            r, c = m.latlon_to_pixel(fr.truth.lat, fr.truth.lon)
            inside += 0 <= r < m.height and 0 <= c < m.width
        heights = [fr.sensors.get("height") for fr in frames if "height" in fr.sensors]
        s, w, n, e = m.bounds_latlon()
        print(
            f"flight {fid}: {len(frames)} frames, {inside} ground-truth points inside the map\n"
            f"  map {m.width}x{m.height} px, {m.gsd_m():.3f} m/px, CRS {m.crs}, "
            f"bounds S{s:.5f} W{w:.5f} N{n:.5f} E{e:.5f}\n"
            f"  CSV height {min(heights):.0f}-{max(heights):.0f} (sensor data, analysis only)"
        )
        if args.plot:
            _plot_track(flight, Path(args.plot) / f"track_{fid}.jpg")


def _plot_track(flight: Flight, out: Path) -> None:
    import cv2

    m = flight.satellite_map()
    img = m.read(max_size=2048)
    scale = img.shape[1] / m.width
    pts = []
    for fr in flight.frames():
        r, c = m.latlon_to_pixel(fr.truth.lat, fr.truth.lon)
        pts.append((round(c * scale), round(r * scale)))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for a, b in zip(pts, pts[1:]):
        cv2.line(bgr, a, b, (0, 0, 255), 2)
    cv2.circle(bgr, pts[0], 6, (0, 255, 0), -1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"  track plot: {out}")


def cmd_eval(args: argparse.Namespace) -> None:
    if args.method not in baselines.REGISTRY:
        raise SystemExit(f"unknown method {args.method!r}; choose from {sorted(baselines.REGISTRY)}")
    for fid in args.flight or available_flights(args.data):
        flight = Flight(args.data, fid)
        localizer = baselines.REGISTRY[args.method]()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = Path(args.out) / f"{stamp}_{args.method}_{fid}"
        summary, _ = evaluate.run(
            localizer,
            flight,
            frames=flight.frames(step=args.step, limit=args.limit),
            image_max_size=args.image_max_size,
            out_dir=out_dir,
        )
        print(evaluate.format_summary(summary))
        print(f"  results: {out_dir}")


def cmd_repeat(args: argparse.Namespace) -> None:
    from . import repeatability

    for fid in args.flight or available_flights(args.data):
        flight = Flight(args.data, fid)
        for name in args.extractor:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = Path(args.out) / f"{stamp}_repeat_{name}_{fid}"
            s = repeatability.run(flight, name, args.step, args.gsd, out_dir, Path(args.out) / "cache",
                                  n_viz=args.viz, limit=args.limit)
            print(repeatability.format_summary(s))
            print(f"  results: {out_dir}")


def cmd_motion(args: argparse.Namespace) -> None:
    from . import motion_eval

    flights = [Flight(args.data, f) for f in (args.flight or available_flights(args.data))]
    out_dir = Path(args.out) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_motion"
    s = motion_eval.run(flights, args.step, out_dir, Path(args.out) / "cache")
    print(motion_eval.format_summary(s))
    print(f"  results: {out_dir}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="gpsloc")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="UAV-VisLoc root folder")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="sanity-check dataset flights")
    c.add_argument("--flight", action="append", help="flight id, e.g. 03 (repeatable; default all)")
    c.add_argument("--plot", metavar="DIR", help="write a ground-truth track over the map to DIR")
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("eval", help="score a localization method")
    e.add_argument("--flight", action="append", help="flight id (repeatable; default all)")
    e.add_argument("--method", default="map-center")
    e.add_argument("--step", type=int, default=1, help="use every Nth frame")
    e.add_argument("--limit", type=int, help="max frames per flight")
    e.add_argument("--image-max-size", type=int, default=1024, help="downscale drone images (px)")
    e.add_argument("--out", default="runs", help="output folder")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("repeat", help="landmark repeatability between drone and map (analysis)")
    r.add_argument("--flight", action="append", help="flight id (repeatable; default all)")
    r.add_argument("--extractor", action="append", default=None,
                   help="corners, lsd or roads (repeatable; default all)")
    r.add_argument("--step", type=int, default=8, help="use every Nth frame")
    r.add_argument("--limit", type=int, help="max aligned frames per flight")
    r.add_argument("--gsd", type=float, default=1.0, help="working resolution, meters per pixel")
    r.add_argument("--viz", type=int, default=4, help="number of frames to draw")
    r.add_argument("--out", default="runs", help="output folder")
    r.set_defaults(func=cmd_repeat)

    mo = sub.add_parser("motion", help="frame-to-frame motion and tilt vs CSV (analysis)")
    mo.add_argument("--flight", action="append", help="flight id (repeatable; default all)")
    mo.add_argument("--step", type=int, default=8, help="use every Nth frame (and its successor)")
    mo.add_argument("--out", default="runs", help="output folder")
    mo.set_defaults(func=cmd_motion)

    args = p.parse_args(argv)
    if getattr(args, "cmd", None) == "repeat" and not args.extractor:
        from .landmarks import EXTRACTORS

        args.extractor = list(EXTRACTORS)
    args.func(args)


if __name__ == "__main__":
    main()
