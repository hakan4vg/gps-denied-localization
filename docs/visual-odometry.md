# Visual odometry: motion, tilt and nadir correction from frame pairs

Date: 2026-11-02. Flights 03 and 04, frame k = every 8th frame, paired with frame k+1
(7 s apart, 95-111 m apart, 65-80% overlap). 130 of 131 pairs estimated.

**Question:** can consecutive frames alone (no GPS, IMU or altimeter) give direction, speed and
camera tilt, and does knowing the tilt remove the ~20 m gap between the image center and the
drone's position found in milestone 2?

**Answer: yes.** Localizing the pixel directly below the camera instead of the image center
cuts the median position error against the CSV from 18-25 m to about 8-9 m.

## Method (`gpsloc/motion.py`, camera only)

Seen from ~500 m the ground is close to a plane, so features matched between frames k and k+1
fit a homography H. With the camera intrinsics K (focal length ≈ 3850 px from milestone 2,
principal point at the image center) it decomposes into:

- **R**: rotation between the frames (heading change);
- **t/d**: camera translation divided by the distance to the ground: direction of travel and
  distance in "heights";
- **n**: the ground-plane normal in the camera's frame. Its angle to the optical axis is the
  camera tilt, and **K·n is the pixel directly below the camera** (nadir).

The decomposition has two physically valid solutions; the wrong one has its normal along the
direction of travel (nearly horizontal in level flight), so the solution with the normal closest
to the optical axis is kept. Synthetic tests (`tests/test_motion.py`) render a flat textured
ground from known poses: tilt is recovered within 0.3°, the nadir pixel within 6 px and motion
within 1%.

## Results (`gpsloc motion --flight 03 --flight 04 --step 8`)

CSV values are used only as ground truth here.

| | Flight 03 | Flight 04 |
|---|---|---|
| Pairs estimated | 67 / 67 | 63 / 64 |
| **Position error vs CSV, image center** (median / p90) | 18.0 / 29.1 m | 24.9 / 47.1 m |
| **Position error vs CSV, nadir pixel** (median) | 9.4 m | 8.6 m |
| Nadir pixel + constant offset calibrated on the *other* flight (median / p90) | **8.8 / 14.9 m** | **8.1 / 18.7 m** |
| Speed error, raw (median) | 7.1% (always high) | 6.1% (always high) |
| Speed error after scale calibrated on the other flight (median / p90) | 2.0% / 7.6% | 2.9% / 6.6% |
| Direction of travel error (median) | 4.0° | 1.6° |
| Tilt about image x vs CSV Omega: correlation / residual | 0.78 / 0.83° | 0.80 / 0.90° |
| Tilt about image y vs CSV Kappa: correlation / residual | 0.85 / 0.60° | 0.95 / 0.80° |
| Constant offset, tilt y − Kappa | 1.62° | 1.58° |
| Heading change vs CSV ΔPhi1: correlation | −0.96 | −0.88 |

(Heading change correlates negatively only because image rotation and `Phi1` turn in opposite
directions.)

## What this means

- **Tilt can be measured from the camera alone.** It agrees with the CSV pitch/roll to about
  0.6-0.9° (some of that is noise in the CSV angles themselves). At 500 m, 1° is ~9 m on the
  ground, which matches the ~8-9 m that remains.
- **The constant 17 m offset from milestone 2 is a camera mounting angle.** Vision tilt about
  image y is 1.6° away from the CSV roll on *both* flights: the camera is mounted ~1.6° off the
  IMU axis. The CSV angles cannot see this; the camera does.
- **Speed and altitude come as a pair.** One camera measures distance traveled *in heights*
  (t/d), so frame timing alone gives speed only once the height is known, or height once the
  speed is known. Without sensors, height comes from the map match: the matched scale gives the
  ground size of a drone pixel, and with the focal length, `height = focal × ground pixel size`.
  Speed then follows from t/d × height / Δt.
- **The ~6-7% speed bias is the same on both flights**, so it is a calibration constant, most
  likely a small error in the focal length estimate (or the CSV height reference). Calibrated on
  one flight, the other flight's speed is within 2-3% (median).

## How to use it in the localizer

1. Map match each frame (still to be built) → image-to-map transform.
2. Visual odometry with the previous frame → nadir pixel. Report the nadir pixel's map position
   as the drone position, not the image center.
3. Matched scale + focal length → altitude; t/d × altitude / Δt → speed; R → heading change.
4. Track the frame-to-frame motion to combine evidence over several frames (milestone 6), and to
   stitch frames into larger strips for matching.

## Limits

- Needs consecutive overlapping frames (the first frame of a flight has no nadir estimate).
- Assumes flat ground; hills and tall buildings bias the plane fit.
- Focal length was itself estimated from the CSV height in milestone 2. A camera spec sheet or a
  one-off calibration would remove that dependence.
- Lens distortion is not modeled.
