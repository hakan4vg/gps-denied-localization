"""Scoring harness: run a localizer over a flight and compare against CSV ground truth.

Metrics: error in meters, success rate within 25/50/100 m, time per image, map index size,
and how often the method answered ok / unsure / failed.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .dataset import Flight, Frame
from .geo import haversine_m
from .localizer import Fix, Localizer, Status

THRESHOLDS_M = (25.0, 50.0, 100.0)


@dataclass
class FrameResult:
    flight_id: str
    filename: str
    status: str
    error_m: float | None
    seconds: float
    confidence: float | None
    pred_lat: float | None
    pred_lon: float | None
    true_lat: float
    true_lon: float
    note: str = ""


@dataclass
class Summary:
    method: str
    flight_id: str
    n_frames: int
    n_ok: int
    n_unsure: int
    n_failed: int
    median_error_m: float | None
    mean_error_m: float | None
    success_rate: dict[str, float]
    success_rate_ok_only: dict[str, float]
    mean_seconds: float
    prepare_seconds: float
    index_size_bytes: int
    extra: dict = field(default_factory=dict)


def score_fix(frame: Frame, fix: Fix, seconds: float) -> FrameResult:
    error = None
    if fix.status != Status.FAILED and fix.lat is not None and fix.lon is not None:
        error = haversine_m(fix.lat, fix.lon, frame.truth.lat, frame.truth.lon)
    return FrameResult(
        flight_id=frame.flight_id,
        filename=frame.image_path.name,
        status=fix.status.value,
        error_m=error,
        seconds=seconds,
        confidence=fix.confidence,
        pred_lat=fix.lat,
        pred_lon=fix.lon,
        true_lat=frame.truth.lat,
        true_lon=frame.truth.lon,
        note=fix.note,
    )


def summarize(
    results: list[FrameResult], method: str, flight_id: str, prepare_seconds: float, index_size: int
) -> Summary:
    n = len(results)
    errors = [r.error_m for r in results if r.error_m is not None]
    ok_errors = [r.error_m for r in results if r.status == Status.OK.value and r.error_m is not None]

    def rate(errs: list[float], t: float) -> float:
        # Denominator is all frames: a failed or "unsure" frame counts as a miss.
        return sum(e <= t for e in errs) / n if n else 0.0

    return Summary(
        method=method,
        flight_id=flight_id,
        n_frames=n,
        n_ok=sum(r.status == Status.OK.value for r in results),
        n_unsure=sum(r.status == Status.UNSURE.value for r in results),
        n_failed=sum(r.status == Status.FAILED.value for r in results),
        median_error_m=float(np.median(errors)) if errors else None,
        mean_error_m=float(np.mean(errors)) if errors else None,
        success_rate={f"{t:g}m": rate(errors, t) for t in THRESHOLDS_M},
        success_rate_ok_only={f"{t:g}m": rate(ok_errors, t) for t in THRESHOLDS_M},
        mean_seconds=float(np.mean([r.seconds for r in results])) if results else math.nan,
        prepare_seconds=prepare_seconds,
        index_size_bytes=index_size,
    )


def run(
    localizer: Localizer,
    flight: Flight,
    frames: Iterable[Frame] | None = None,
    image_max_size: int | None = None,
    out_dir: Path | None = None,
    progress: bool = True,
) -> tuple[Summary, list[FrameResult]]:
    sat_map = flight.satellite_map()
    t0 = time.perf_counter()
    localizer.prepare(sat_map)
    prepare_s = time.perf_counter() - t0

    frames = list(flight.frames() if frames is None else frames)
    results: list[FrameResult] = []
    for i, frame in enumerate(frames, 1):
        image = frame.load_image(image_max_size)  # loading time is not counted
        t = time.perf_counter()
        try:
            fix = localizer.localize(image)
        except Exception as exc:  # a crash on one frame is a failed frame, not a failed run
            fix = Fix.failed(f"{type(exc).__name__}: {exc}")
        results.append(score_fix(frame, fix, time.perf_counter() - t))
        if progress and (i % 25 == 0 or i == len(frames)):
            r = results[-1]
            err = f"{r.error_m:.0f} m" if r.error_m is not None else "-"
            print(f"[{localizer.name}] {i}/{len(frames)} last={r.filename} {r.status} {err}", flush=True)

    summary = summarize(results, localizer.name, flight.flight_id, prepare_s, localizer.index_size_bytes())
    if out_dir is not None:
        write_outputs(out_dir, summary, results)
    return summary, results


def write_outputs(out_dir: Path, summary: Summary, results: list[FrameResult]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
    with (out_dir / "frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else ["filename"])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def format_summary(s: Summary) -> str:
    def fmt(v: float | None) -> str:
        return "-" if v is None else f"{v:,.1f}"

    rates = "  ".join(f"<{k}: {v:6.1%}" for k, v in s.success_rate.items())
    return (
        f"{s.method} on flight {s.flight_id}: {s.n_frames} frames "
        f"(ok {s.n_ok}, unsure {s.n_unsure}, failed {s.n_failed})\n"
        f"  error median {fmt(s.median_error_m)} m, mean {fmt(s.mean_error_m)} m\n"
        f"  success {rates}\n"
        f"  {s.mean_seconds * 1000:.1f} ms/image, prepare {s.prepare_seconds:.1f} s, "
        f"index {s.index_size_bytes / 1e6:.2f} MB"
    )
