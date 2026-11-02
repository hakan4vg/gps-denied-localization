# Error budget of the nadir correction, and what reduces it

Date: 2026-11-03. Flights 03, 04 (flat, tuned on), 06 and 11 (hilly, held out), every 8th
frame. Flight 05 still does not align (satellite image from another year).
Data: `runs/20261102-231416_vocheck`. Scripts: `scripts/analysis/`, model in
`gpsloc/error_budget.py`.

**Question:** the vision nadir pixel cut the error from ~20 m to ~9 m on flights 03/04. What is
the remaining 9 m made of, does the result hold on unseen hilly flights, and what lowers it?

**Answer:** vision tilt plus a DEM slope correction gives **7-9 m median on three of four
flights, including held-out flight 11**, with the camera mount angle calibrated on *other*
flights. The CSV attitude does not transfer between flights, so vision tilt is the better
source, not just a replacement. Rough terrain (flight 06) breaks the flat-ground model, and a DEM
roughness gate detects it in advance. What remains is mostly vision-tilt noise (~0.4-0.7°).

All errors are against the CSV position, through the reference alignment, which is itself
seeded from CSV position, heading and height. This measures the nadir correction, not a
localizer (milestone 3).

## Position error, median / p90 m, mount angle calibrated on the other flights

| Method | 03 | 04 | 06 (rough) | 11 (hilly, held out) |
|---|---|---|---|---|
| Image center (raw) | 18.0 / 29.1 | 24.9 / 47.1 | 16.9 / 33.4 | 22.6 / 41.7 |
| CSV tilt | 23.8 / 29.6 | 32.3 / 42.3 | 8.6 / 15.5 | 55.7 / 65.7 |
| CSV tilt, calibrated on the *same* flight | 6.7 / 12.0 | 8.6 / 20.3 | 9.0 / 16.8 | 10.5 / 19.6 |
| Vision tilt (SIFT, k±1) | 8.4 / 16.8 | 7.2 / 13.3 | 29.2 / 56.3 | 25.7 / 40.0 |
| **Vision tilt + DEM slope correction** | 9.3 / 14.1 | 7.4 / 14.3 | 26.7 / 42.4 | **7.7 / 17.4** |
| Same, Lucas-Kanade 480 px (≈25 ms/fix) | 9.7 / 16.2 | 8.4 / 14.1 | (53% avail.) | 8.0 / 16.5 |

## Findings

1. **The DEM slope correction is what makes hilly ground work.** The homography gives the ground
   plane's normal, but the drone's position is along the true vertical. On flight 11 (slope ~2°,
   770 m above ground) the difference is 25 m → 8 m. On flat flights 03/04 the correction is
   neutral to slightly worse (DEM noise on 0.2° slopes), so apply it only above ~1° slope.

2. **The CSV attitude does not carry over between flights; vision does.** The constant offset
   between camera and CSV tilt is +134 / +140 px (≈2.0°) on flights 03/04 but −153 px on
   flight 11: a different airframe, IMU mounting or angle convention. Calibrated within a flight,
   CSV tilt is good (6.7-10.5 m), but you can't know that calibration in advance. The vision mount
   offset is the same on every flight (≈ +30 px, 0.45°), so one factory calibration is enough.
   Blending vision and CSV tilt never beat vision alone out of sample for this reason.

3. **Tilt noise per source, without ground truth (three-cornered hat, deg):**

   | | CSV | vision (k, k+1) | vision (k, k+2) |
   |---|---|---|---|
   | 03 | 0.54 / 0.32 | 0.44 / 0.50 | 0.86 / 1.34 |
   | 04 | 0.80 / 0.72 | 0.37 / 0.41 | 0.63 / 1.99 |
   | 11 | 1.78 / 1.83 | 0.68 / ~0 | 0.88 / 1.00 |

   (x / y axis. Both vision estimates share frame k, so their common error lands in the CSV
   column: the CSV figures are upper bounds.) Vision with the next frame is as good as or better
   than the CSV everywhere; on flight 11 the CSV is ~3× noisier. The k+2 pair (less overlap) is
   worse. At 460-770 m, 0.4-0.7° is 3-9 m: **tilt noise is now the floor.**

4. **CSV errors peak when attitude changes quickly; vision errors don't.** With the CSV tilt, the
   worst frames are the ones where Omega/Kappa change fast between neighbouring frames
   (rank correlation +0.27-0.29, ~0 against tilt size), mostly across-track (roll). This fits CSV
   attitude sampled at a different moment from the exposure. Vision error is the same in calm and
   fast halves (03: 9.4 vs 9.3 m, 11: 8.4 vs 7.0 m).

5. **Rough terrain breaks the flat-ground model, and the DEM predicts it.** Flight 06's footprint
   relief (DEM sd around the best-fit plane / height above ground) is 2.6%, against 0.16-0.39% on
   the others. Gating on it (vision + slope correction, all flights):

   | Keep frames with relief < | Availability | Median | p90 | Off by > 30 m |
   |---|---|---|---|---|
   | 0.5% | 81% | 8.1 m | 14.3 m | 0.0% |
   | **1%** | **88%** | **8.1 m** | **14.9 m** | **0.0%** |
   | 2% | 91% | 8.1 m | 15.3 m | 1.6% |
   | no gate | 100% | 8.6 m | 21.4 m | 5.9% |

   A 1% gate answers "unsure" for all of flight 06 and almost nothing else. Recovering flight 06
   needs a model that isn't a plane: pose from map correspondences with DEM heights (PnP).

6. **Map georeferencing offset: small.** Fitting a constant north/east offset per map (it rotates
   in the image as the drone turns, so it separates from the mount offset) finds ~3-4 m north on
   03/04 and gains ~1 m. Worth keeping in mind; not the main term.

7. **Photo/GPS timestamp lag can't be measured from this data.** The camera is fixed to the
   airframe, so the direction of travel in image axes barely changes (±2-12°), and lag × velocity
   is indistinguishable from the mount offset. Fits flip sign between flights (−0.4 to +0.7 s).
   Whatever lag exists is folded into the mount calibration at typical speed; it would show up
   as error at very different speeds. A real flight log with IMU timing would settle it.

8. **Speed with height from the map scale (no CSV height) is within 2.7-4.5% median, bias +1.3 to
   +3.3%.** The registration scale gives a height 0.96 × the DEM height above ground on
   03/04 (sd 1.5%). The same ~3-4% scale constant shows up in both: most likely the 3850 px focal
   length (estimated, not from a spec sheet), possibly a geoid/ellipsoid height offset.

9. **Altitude from terrain parallax doesn't work at these heights.** `gpsloc relief`
   (`runs/20261103-000718_relief`) gave an estimate for only 9 of ~190 frames, all too low:

   | Flight | Frames | Estimated | Median error |
   |---|---|---|---|
   | 03 | 63 | 1 | −112% |
   | 04 | 60 | 0 | – |
   | 06 | 6 | 4 | −24% |
   | 11 | 60 | 4 | −29% |

   Relief within a footprint is only 5-7 m sd at 300-770 m height above ground, too little
   parallax to measure. Height should come from the map-match scale (finding 8, within ~3%).

## What to do next, by expected gain

1. **Wire it into the localizer:** vision nadir (Lucas-Kanade 480 px, k±1), DEM slope correction
   above ~1° slope, one fixed mount offset, and a DEM roughness gate that returns "unsure".
2. **Lower tilt noise, the current floor:** combine more neighbours (k−2…k+2) with a robust mean,
   or filter tilt over time. Halving 0.5° takes ~4 m off at these heights.
3. **Rough terrain (flight 06):** camera pose from map correspondences plus DEM heights (PnP) in
   place of the plane fit. The reference alignment is affine, so it carries no tilt today and
   would need a homography/PnP refinement at higher resolution.
4. **Calibrate the focal length** (camera spec or checkerboard) to remove the 3-4% scale/speed
   bias.
5. **Real flight log with IMU and camera timing** to measure the photo/GPS lag and run the
   fault-injection tests.

## Fixes made along the way

- Flight 06 frames 93-98 have a CSV altitude ~340 m too low for six frames (6 m below the DEM
  ground). Alignment crashed on the negative image scale. `dem.usable_agl()` now rejects heights
  under 20 m above ground; alignment, wind and validation all use it.
