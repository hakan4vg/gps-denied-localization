"""Reference alignment of a drone frame to the map, FOR ANALYSIS ONLY.

Uses ground truth (position) and sensor data (heading Phi1, height) to start from a close guess,
then refines with SIFT matches restricted to that neighborhood and a RANSAC homography. The
result is the "true" drone-pixel -> map-pixel mapping used to measure landmark repeatability.
A localizer must never import this module.

Camera model: ground size of a drone pixel = height_m / FOCAL_PX. FOCAL_PX was estimated from
frames that registered without any prior (see docs/notes); it is a camera intrinsic, not a sensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .dataset import Frame
from .geo import SatelliteMap, haversine_m

FOCAL_PX = 3850.0  # full-resolution drone pixels (3976 px wide images)
WORK_GSD = 0.6  # meters per pixel used for matching


@dataclass
class Alignment:
    ok: bool
    H: np.ndarray | None  # 3x3, full-res drone (x, y) -> full-res map (col, row)
    inliers: int
    scale: float | None = None  # map px per drone px
    rotation_deg: float | None = None
    center_offset_m: float | None = None  # registered image center vs CSV lat/lon
    center_offset_ne: tuple[float, float] | None = None  # (north, east) meters
    note: str = ""

    def drone_to_map(self, pts_xy: np.ndarray) -> np.ndarray:
        """Nx2 drone (x, y) full-res -> Nx2 map (col, row) full-res."""
        p = cv2.perspectiveTransform(np.asarray(pts_xy, np.float64).reshape(-1, 1, 2), self.H)
        return p.reshape(-1, 2)


def _similarity(scale: float, angle_deg: float, src_center, dst_center) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T1 = np.array([[1, 0, -src_center[0]], [0, 1, -src_center[1]], [0, 0, 1]])
    T2 = np.array([[1, 0, dst_center[0]], [0, 1, dst_center[1]], [0, 0, 1]])
    return T2 @ R @ T1


_sift = None


def _detect(gray: np.ndarray):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    kp, desc = _sift.detectAndCompute(gray, None)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, desc


def align_frame(
    frame: Frame,
    sat_map: SatelliteMap,
    image: np.ndarray | None = None,
    search_radius_m: float = 80.0,
    min_inliers: int = 15,
) -> Alignment:
    from .dem import usable_agl

    # CSV height is above sea level; use height above ground where a DEM tile is available.
    height = usable_agl(frame)
    heading = frame.sensors.get("Phi1")
    if height is None or heading is None:
        return Alignment(False, None, 0, note="missing or implausible height/heading")
    img = frame.load_image() if image is None else image
    gsd_map = sat_map.gsd_m()
    gsd_drone = height / FOCAL_PX

    # Work at WORK_GSD on both sides.
    kd = gsd_drone / WORK_GSD
    km = gsd_map / WORK_GSD
    small = cv2.resize(img, None, fx=kd, fy=kd, interpolation=cv2.INTER_AREA)

    r, c = sat_map.latlon_to_pixel(frame.truth.lat, frame.truth.lon)
    half = int((0.5 * math.hypot(*img.shape[:2]) * gsd_drone + 2 * search_radius_m) / gsd_map)
    r0, c0 = max(0, int(r) - half), max(0, int(c) - half)
    h = min(sat_map.height - r0, 2 * half)
    w = min(sat_map.width - c0, 2 * half)
    crop = sat_map.read(r0, c0, h, w, max_size=max(1, round(max(h, w) * km)))
    sy, sx = crop.shape[0] / h, crop.shape[1] / w  # crop px per map px

    g1 = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    p1, d1 = _detect(g1)
    p2, d2 = _detect(g2)
    if d1 is None or d2 is None or len(p1) < 8 or len(p2) < 8:
        return Alignment(False, None, 0, note="too few keypoints")

    # Prior in working coordinates: small drone px -> crop px.
    prior = _similarity(
        1.0, heading, (g1.shape[1] / 2, g1.shape[0] / 2), ((c - c0) * sx, (r - r0) * sy)
    )
    pred = cv2.perspectiveTransform(p1.reshape(-1, 1, 2).astype(np.float64), prior).reshape(-1, 2)

    # Match each drone keypoint only against map keypoints near its predicted spot.
    from scipy.spatial import cKDTree

    tree = cKDTree(p2)
    radius = search_radius_m / WORK_GSD
    src, dst = [], []
    for i, cand in enumerate(tree.query_ball_point(pred, radius)):
        if len(cand) < 1:
            continue
        dist = np.linalg.norm(d2[cand] - d1[i], axis=1)
        order = np.argsort(dist)
        if len(order) > 1 and dist[order[0]] > 0.9 * dist[order[1]]:
            continue
        src.append(p1[i])
        dst.append(p2[cand[order[0]]])
    if len(src) < min_inliers:
        return Alignment(False, None, len(src), note=f"only {len(src)} candidate matches")
    src, dst = np.float32(src), np.float32(dst)
    # Near-nadir view of mostly flat ground: an affine model is enough and much more stable
    # with few, noisy matches than a full homography.
    Aw, mask = cv2.estimateAffine2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=0.999
    )
    n_in = int(mask.sum()) if mask is not None else 0
    if Aw is None or n_in < min_inliers:
        return Alignment(False, None, n_in, note=f"{n_in} inliers of {len(src)}")
    # Refine on inliers only.
    keep = mask.ravel().astype(bool)
    Aw, _ = cv2.estimateAffine2D(src[keep], dst[keep], method=cv2.LMEDS)
    Hw = np.vstack([Aw, [0, 0, 1]])

    # Lift to full resolution: drone full -> small -> crop -> map full.
    D = np.diag([kd, kd, 1.0])
    M = np.array([[1 / sx, 0, c0], [0, 1 / sy, r0], [0, 0, 1]])
    H = M @ Hw @ D
    H /= H[2, 2]

    A = H[:2, :2]
    scale = math.sqrt(abs(np.linalg.det(A)))
    rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    expected = gsd_drone / gsd_map
    rot_err = (rot - heading + 180) % 360 - 180
    if not (0.75 < scale / expected < 1.33) or abs(rot_err) > 20:
        return Alignment(
            False, None, n_in, scale, rot,
            note=f"disagrees with prior: scale x{scale / expected:.2f}, rotation {rot_err:+.0f} deg",
        )

    center = cv2.perspectiveTransform(
        np.array([[[img.shape[1] / 2, img.shape[0] / 2]]], np.float64), H
    )[0, 0]
    lat, lon = sat_map.pixel_to_latlon(center[1], center[0])
    north = haversine_m(frame.truth.lat, lon, lat, lon) * np.sign(lat - frame.truth.lat)
    east = haversine_m(lat, frame.truth.lon, lat, lon) * np.sign(lon - frame.truth.lon)
    return Alignment(
        True, H, n_in, scale, rot,
        center_offset_m=haversine_m(lat, lon, frame.truth.lat, frame.truth.lon),
        center_offset_ne=(float(north), float(east)),
    )
