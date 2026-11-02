"""Synthetic check of motion.estimate_pair: render a flat textured ground from two known poses."""

import math

import cv2
import numpy as np
import pytest

from gpsloc.motion import estimate_pair, intrinsics

W, H_IMG, F = 1000, 700, 970.0  # ~55 deg horizontal field of view, like the real camera
GROUND_PX_PER_M = 2.0


def _rot(axis: str, deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


# World: X east, Y north, Z up, ground at Z = 0. A nadir camera with image x = east and
# image y = south has camera axes x_c = X, y_c = -Y, z_c = -Z.
NADIR = np.diag([1.0, -1.0, -1.0])


def _camera(center, pitch=0.0, roll=0.0, yaw=0.0):
    R = _rot("x", pitch) @ _rot("y", roll) @ _rot("z", yaw) @ NADIR  # world -> camera
    return R, np.asarray(center, float)


def _texture(size=2400, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 255, (size // 8, size // 8), dtype=np.uint8)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (0, 0), 2)
    for _ in range(600):  # add sharp shapes for good keypoints
        x, y = rng.integers(0, size, 2)
        cv2.rectangle(g, (int(x), int(y)), (int(x + rng.integers(5, 40)), int(y + rng.integers(5, 40))),
                      int(rng.integers(0, 255)), -1)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


def _render(tex, R, C, K):
    # ground pixel (u, v) -> world (X, Y, 0); center of texture at world origin, v grows south
    size = tex.shape[0]
    A = np.array([[1 / GROUND_PX_PER_M, 0, -size / 2 / GROUND_PX_PER_M],
                  [0, -1 / GROUND_PX_PER_M, size / 2 / GROUND_PX_PER_M],
                  [0, 0, 1]])
    t = -R @ C
    Hg = K @ np.column_stack([R[:, 0], R[:, 1], t])  # world plane (X, Y, 1) -> image
    return cv2.warpPerspective(tex, Hg @ A, (W, H_IMG), flags=cv2.INTER_LINEAR)


@pytest.mark.parametrize("pitch,roll,yaw", [(0, 0, 0), (3, -2, 0), (-4, 1.5, 30)])
def test_recovers_tilt_nadir_and_motion(pitch, roll, yaw):
    K = intrinsics(W, H_IMG, F)
    tex = _texture()
    h = 450.0
    R1, C1 = _camera([0, 0, h], pitch, roll, yaw)
    R2, C2 = _camera([60, 70, h], pitch + 0.5, roll - 0.3, yaw + 2)
    m = estimate_pair(_render(tex, R1, C1, K), _render(tex, R2, C2, K), K, work_width=W)
    assert m.ok, m.note

    true_n = R1 @ np.array([0, 0, -1.0])  # direction from camera toward the ground, camera 1 axes
    assert math.degrees(math.acos(np.clip(m.n @ true_n, -1, 1))) < 0.3

    true_nadir = (K @ true_n)[:2] / (K @ true_n)[2]
    assert np.linalg.norm(m.nadir_px(K) - true_nadir) < 6  # pixels (~3 m at this scale)

    true_motion = R1 @ (C2 - C1) / h  # camera-1 axes, in heights
    assert np.allclose(m.motion_cam_k, true_motion, atol=0.01)
    assert m.baseline_ratio == pytest.approx(math.hypot(60, 70) / h, rel=0.03)
