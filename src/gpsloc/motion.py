"""Camera motion and attitude from two overlapping frames (camera only, no sensors).

The ground seen from ~500 m is close to a plane, so matched features between frames k and k+1
fit a homography H. With known intrinsics K it decomposes as

    K^-1 H K  ~  R + t n^T / d

R:   rotation from camera k to camera k+1 (heading change, small tilt changes)
t/d: translation over the distance to the ground plane: direction of travel, and speed in
     "heights per second" (meters need the height, e.g. from the map-match scale)
n:   ground-plane normal in camera k's frame. Its angle to the optical axis is the camera's
     tilt relative to the ground, and K n is the pixel directly below the camera (nadir).

Localizing the nadir pixel instead of the image center removes the tilt-induced offset between
"where the camera looks" and "where the drone is".

Limits: assumes flat ground and a pinhole camera with the principal point at the image center
(no lens-distortion model yet). Needs enough baseline: t/d of ~0.2 here (95 m at 470 m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FOCAL_PX = 3850.0  # estimated in milestone 2 for the UAV-VisLoc camera (3976 x 2652 images)


def intrinsics(width: int, height: int, focal_px: float = FOCAL_PX) -> np.ndarray:
    return np.array([[focal_px, 0, width / 2], [0, focal_px, height / 2], [0, 0, 1.0]])


@dataclass
class PairMotion:
    ok: bool
    inliers: int = 0
    H: np.ndarray | None = None  # full-res frame k -> frame k+1 pixels
    R: np.ndarray | None = None
    t_over_d: np.ndarray | None = None  # in camera k+1 frame
    n: np.ndarray | None = None  # ground normal in camera k frame, n_z > 0
    note: str = ""

    @property
    def tilt_deg(self) -> float:
        return math.degrees(math.acos(min(1.0, float(self.n[2]))))

    def nadir_px(self, K: np.ndarray) -> np.ndarray:
        """Pixel (x, y) in frame k directly below the camera."""
        p = K @ self.n
        return p[:2] / p[2]

    @property
    def yaw_change_deg(self) -> float:
        """Rotation about the optical axis from frame k to k+1 (image-plane rotation)."""
        return math.degrees(math.atan2(self.R[1, 0], self.R[0, 0]))

    @property
    def motion_cam_k(self) -> np.ndarray:
        """Camera center displacement, in camera k axes, in units of the ground distance d."""
        return -self.R.T @ self.t_over_d

    @property
    def travel_dir_img_deg(self) -> float:
        """Direction of travel as an angle in frame k's image (x right, y down)."""
        m = self.motion_cam_k
        return math.degrees(math.atan2(m[1], m[0]))

    @property
    def baseline_ratio(self) -> float:
        """Horizontal distance traveled / height."""
        m = self.motion_cam_k
        return float(math.hypot(m[0], m[1]))


_sift = None


def _features(gray: np.ndarray):
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=8000)
    return _sift.detectAndCompute(gray, None)


def estimate_pair(
    img1: np.ndarray, img2: np.ndarray, K: np.ndarray, work_width: int = 1600, min_inliers: int = 50
) -> PairMotion:
    """Estimate motion between two full-resolution RGB frames of the same camera."""
    s = work_width / img1.shape[1]
    g1 = cv2.cvtColor(cv2.resize(img1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(cv2.resize(img2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    k1, d1 = _features(g1)
    k2, d2 = _features(g2)
    if d1 is None or d2 is None or len(k1) < min_inliers or len(k2) < min_inliers:
        return PairMotion(False, note="too few features")
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n2 in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n2.distance]
    if len(good) < min_inliers:
        return PairMotion(False, note=f"{len(good)} matches")
    p1 = np.float64([k1[m.queryIdx].pt for m in good]) / s
    p2 = np.float64([k2[m.trainIdx].pt for m in good]) / s
    H, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 2.0 / s, maxIters=10000, confidence=0.9999)
    if H is None:
        return PairMotion(False, note="no homography")
    inl = mask.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return PairMotion(False, int(inl.sum()), note="few inliers")
    # Refine on inliers with least squares (Levenberg-Marquardt inside findHomography).
    H, _ = cv2.findHomography(p1[inl], p2[inl], 0)
    H /= H[2, 2]

    n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    Kinv = np.linalg.inv(K)
    rays = (Kinv @ np.column_stack([p1[inl], np.ones(inl.sum())]).T)  # 3 x N, camera k
    best = None
    for R, t, n in zip(Rs, ts, ns):
        n = n.ravel()
        t = t.ravel()
        if n[2] < 0:  # same plane, opposite normal sign: flip n and t together
            n, t = -n, -t
        # Physically valid: every inlier point lies in front of camera k (n . ray > 0) and of
        # camera k+1 (depth of R X + t > 0, with X = ray * d / (n . ray), d = 1).
        depth1 = n @ rays
        if np.mean(depth1 > 0) < 0.99:
            continue
        X = rays / depth1
        X2 = R @ X + t[:, None]
        if np.mean(X2[2] > 0) < 0.99:
            continue
        # Of the (usually two) valid solutions, the wrong one has n close to the direction of
        # travel, which is nearly horizontal for a level flight: keep the one closest to nadir.
        if best is None or n[2] > best[2][2]:
            best = (R, t, n)
    if best is None:
        return PairMotion(False, int(inl.sum()), H, note="no valid decomposition")
    R, t, n = best
    return PairMotion(True, int(inl.sum()), H, R, t, n / np.linalg.norm(n))
