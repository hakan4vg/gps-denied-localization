# gpsloc: GPS-denied drone localization

Find a drone's position from a **downward camera image** and a **satellite map stored on the
drone**: no GPS, compass, altimeter, IMU or last known position. Later: heading from the
recovered rotation, altitude from the recovered scale, speed from consecutive fixes.

Planned method: "lost-in-space" geometric hashing, the trick star trackers and
Astrometry.net use. Groups of 4 landmark points (starting with road intersections) are turned
into codes that don't change with rotation, scale or position, so the drone image can be
looked up in a precomputed map index instead of searched over every angle and altitude.

Target hardware is CPU-only (Raspberry Pi 5 / Jetson class).

## Status

- [x] Milestone 1: dataset loader, scoring harness, trivial baselines
- [x] Milestone 2: landmark extraction and drone/satellite repeatability (go/no-go):
  **no-go for road-intersection quads**, see [docs/milestone2-repeatability.md](docs/milestone2-repeatability.md)
- [x] Visual odometry from frame pairs: tilt, speed, direction; nadir correction cuts position
  error from ~20 m to ~8-9 m, see [docs/visual-odometry.md](docs/visual-odometry.md)
- [x] Error budget: vision tilt + DEM slope correction gives 7-9 m on flat and held-out hilly
  flights, CSV attitude does not transfer between flights, DEM roughness gate flags rough terrain,
  see [docs/error-budget.md](docs/error-budget.md)
- [ ] Milestone 3: quad hashing on one flight, no metadata
- [ ] Milestone 4: evaluation across flights and terrain types
- [ ] Milestone 5: heading, altitude, speed
- [ ] Milestone 6: tracking, particle filter, low-texture fallback

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

Data: [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc), unpacked to
`~/data/UAV-VisLoc` (override with `GPSLOC_DATA` or `--data`).

## Usage

```bash
gpsloc check --flight 03 --plot runs/   # ground truth inside map? map resolution? track image
gpsloc eval --flight 03 --method map-center --step 10
```

`eval` writes `summary.json` and per-frame `frames.csv` under `runs/`.

```bash
gpsloc repeat --flight 03 --extractor corners --step 8   # landmark repeatability (analysis)
```

## Rules the harness enforces

- A localizer gets only the satellite map (`prepare`) and the drone image (`localize`).
  Ground-truth lat/lon from the CSV is used only for scoring. Height and angles are sensor data
  and are never passed in.
- Success rates count failed and "unsure" answers as misses. Answering "unsure" is still
  better than a confident wrong answer, and is reported separately.
