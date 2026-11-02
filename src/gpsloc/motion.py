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


@dataclass
class Features:
    pts: np.ndarray  # Nx2 keypoint positions in FULL-resolution pixels
    desc: np.ndarray | None
    scale: float  # working px per full-res px
    detector: str


_detectors: dict[str, cv2.Feature2D] = {}


def detect(img: np.ndarray, work_width: int = 1600, detector: str = "sift", n_features: int = 8000) -> Features:
    """Detect keypoints on a downscaled grayscale copy; positions are returned at full resolution.
    Compute once per frame and reuse for both neighboring pairs."""
    key = f"{detector}:{n_features}"
    if key not in _detectors:
        if detector == "sift":
            _detectors[key] = cv2.SIFT_create(nfeatures=n_features)
        elif detector == "orb":
            _detectors[key] = cv2.ORB_create(nfeatures=n_features, fastThreshold=10)
        else:
            raise ValueError(detector)
    s = work_width / img.shape[1]
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    kp, desc = _detectors[key].detectAndCompute(g, None)
    pts = np.float64([k.pt for k in kp]).reshape(-1, 2) / s
    return Features(pts, desc, s, detector)


def estimate_pair(
    img1: np.ndarray, img2: np.ndarray, K: np.ndarray, work_width: int = 1600, min_inliers: int = 50,
    detector: str = "sift",
) -> PairMotion:
    """Estimate motion between two full-resolution RGB frames of the same camera."""
    return estimate_from_features(
        detect(img1, work_width, detector), detect(img2, work_width, detector), K, min_inliers
    )


def estimate_from_features(f1: Features, f2: Features, K: np.ndarray, min_inliers: int = 50) -> PairMotion:
    """Motion from frame 1 to frame 2. The ground normal is returned in frame 1's camera axes."""
    d1, d2 = f1.desc, f2.desc
    if d1 is None or d2 is None or len(f1.pts) < min_inliers or len(f2.pts) < min_inliers:
        return PairMotion(False, note="too few features")
    if f1.detector == "orb":
        matcher, ratio = cv2.BFMatcher(cv2.NORM_HAMMING), 0.8
    else:
        matcher, ratio = cv2.BFMatcher(), 0.75
    matches = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n2 in (p for p in matches if len(p) == 2) if m.distance < ratio * n2.distance]
    if len(good) < min_inliers:
        return PairMotion(False, note=f"{len(good)} matches")
    p1 = f1.pts[[m.queryIdx for m in good]]
    p2 = f2.pts[[m.trainIdx for m in good]]
    s = f1.scale
    H, mask = cv2.findHomography(p1, p2, cv2.USAC_MAGSAC, 2.0 / s, maxIters=10000, confidence=0.9999)
    if H is None:
        return PairMotion(False, note="no homography")
    inl = mask.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return PairMotion(False, int(inl.sum()), note="few inliers")
    # Refine on inliers with least squares (Levenberg-Marquardt inside findHomography).
    H, _ = cv2.findHomography(p1[inl], p2[inl], 0)
    return _decompose(H / H[2, 2], K, p1[inl])


def _decompose(H: np.ndarray, K: np.ndarray, p1_inl: np.ndarray) -> PairMotion:
    """Pick the physically valid, closest-to-nadir solution of the homography decomposition."""
    n_in = len(p1_inl)
    n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    Kinv = np.linalg.inv(K)
    rays = (Kinv @ np.column_stack([p1_inl, np.ones(n_in)]).T)  # 3 x N, camera k
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
        return PairMotion(False, n_in, H, note="no valid decomposition")
    R, t, n = best
    return PairMotion(True, n_in, H, R, t, n / np.linalg.norm(n))


def to_gray(img: np.ndarray, width: int) -> tuple[np.ndarray, float]:
    """Downscale to `width` and convert to 8-bit gray, as a small camera's ISP would deliver."""
    s = width / img.shape[1]
    g = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return g, s


def global_shift(g1: np.ndarray, g2: np.ndarray, size: int = 128) -> tuple[np.ndarray, float]:
    """Whole-image translation g1 -> g2 by phase correlation on a size x size thumbnail
    (one 128x128 FFT pair: cheap on a Cortex-M7 with CMSIS-DSP). Returns (dx, dy) in g1 pixels
    and the correlation peak strength."""
    h, w = g1.shape
    k = size / max(h, w)
    a = cv2.resize(g1, None, fx=k, fy=k, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(g2, None, fx=k, fy=k, interpolation=cv2.INTER_AREA).astype(np.float32)
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), peak = cv2.phaseCorrelate(a, b, win)
    return np.array([dx, dy]) / k, peak


def lk_track(g1: np.ndarray, g2: np.ndarray, p1: np.ndarray, max_fb_px: float = 0.5):
    """Pyramidal Lucas-Kanade seeded with the global shift, plus a forward-backward check."""
    shift, _ = global_shift(g1, g2)
    lk = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
              flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    p2, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p1, (p1 + shift).astype(np.float32), **lk)
    p1b, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, p2, (p2 - shift).astype(np.float32), **lk)
    good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p1 - p1b, axis=1) < max_fb_px)
    return p2, p1b, good


def track_pair(
    g1: np.ndarray, g2: np.ndarray, s: float, K: np.ndarray, n_points: int = 200, grid: tuple[int, int] = (8, 6),
    min_inliers: int = 30,
) -> PairMotion:
    """Microcontroller-class motion estimate: Shi-Tomasi corners spread over a grid, pyramidal
    Lucas-Kanade tracking with a forward-backward check, RANSAC homography, decomposition.

    No descriptors, no matching search: cost is O(points x patch size x pyramid levels), the same
    family of computation PX4FLOW runs on an STM32F4. g1/g2 are gray frames at scale s of K."""
    h, w = g1.shape
    gx, gy = grid
    per_cell = max(1, n_points // (gx * gy))
    pts = []
    for j in range(gy):
        for i in range(gx):
            x0, x1 = i * w // gx, (i + 1) * w // gx
            y0, y1 = j * h // gy, (j + 1) * h // gy
            c = cv2.goodFeaturesToTrack(g1[y0:y1, x0:x1], per_cell, 0.01, max(3, w // 80), blockSize=5)
            if c is not None:
                pts.append(c.reshape(-1, 2) + (x0, y0))
    if not pts:
        return PairMotion(False, note="no corners")
    p1 = np.concatenate(pts).astype(np.float32)
    p2, p1b, good = lk_track(g1, g2, p1)
    if good.sum() < min_inliers:
        return PairMotion(False, note=f"{good.sum()} tracks")
    a, b = p1[good].astype(np.float64) / s, p2[good].astype(np.float64) / s
    return _from_correspondences(a, b, K, 1.0 / s, min_inliers)


def _from_correspondences(p1, p2, K, thresh_full_px, min_inliers) -> PairMotion:
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, thresh_full_px, maxIters=500, confidence=0.999)
    if H is None:
        return PairMotion(False, note="no homography")
    inl = mask.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return PairMotion(False, int(inl.sum()), note="few inliers")
    H, _ = cv2.findHomography(p1[inl], p2[inl], 0)
    return _decompose(H / H[2, 2], K, p1[inl])


def combine_normals(normals: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Average several estimates of the same frame's ground normal (all in that frame's axes)."""
    w = np.ones(len(normals)) if weights is None else np.asarray(weights, float)
    n = (np.asarray(normals) * w[:, None]).sum(0)
    return n / np.linalg.norm(n)


def nadir_from_normal(n: np.ndarray, K: np.ndarray) -> np.ndarray:
    p = K @ n
    return p[:2] / p[2]
