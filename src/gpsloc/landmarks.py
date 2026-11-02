"""Landmark point extractors.

Every extractor takes an RGB uint8 image already resampled to a known ground resolution
(`gsd` meters per pixel) and returns an Nx2 float array of (x, y) pixel positions. Parameters are
in meters so the same extractor behaves the same on drone and satellite images.

Extractors, cheapest first:
- corners:   Shi-Tomasi corners. Generic baseline, not road-specific.
- lsd:       junctions where long straight line segments meet (roads, field edges, buildings).
- roads:     classic road mask (low saturation, mid brightness, line-shaped) -> skeleton ->
             nodes of degree >= 3.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _clahe(gray: np.ndarray) -> np.ndarray:
    # Equalize local contrast: drone and satellite exposure differ a lot.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def nms_points(pts: np.ndarray, radius_px: float, scores: np.ndarray | None = None) -> np.ndarray:
    """Keep the strongest point within each radius (greedy)."""
    if len(pts) == 0:
        return pts.reshape(0, 2)
    from scipy.spatial import cKDTree

    order = np.argsort(-scores) if scores is not None else np.arange(len(pts))
    tree = cKDTree(pts)
    taken = np.zeros(len(pts), bool)
    keep = []
    for i in order:
        if taken[i]:
            continue
        keep.append(i)
        taken[tree.query_ball_point(pts[i], radius_px)] = True
    return pts[keep]


def cluster_points(pts: np.ndarray, radius_px: float) -> np.ndarray:
    """Merge points closer than radius (single linkage) into their centroids."""
    if len(pts) == 0:
        return pts.reshape(0, 2)
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    graph = cKDTree(pts).sparse_distance_matrix(cKDTree(pts), radius_px, output_type="coo_matrix")
    n, labels = connected_components(graph, directed=False)
    out = np.zeros((n, 2))
    np.add.at(out, labels, pts)
    return out / np.bincount(labels, minlength=n)[:, None]


# --- corners -------------------------------------------------------------------------------
def corners(img: np.ndarray, gsd: float, min_dist_m: float = 8.0, per_km2: int = 400) -> np.ndarray:
    g = _clahe(_gray(img))
    area_km2 = g.shape[0] * g.shape[1] * gsd * gsd / 1e6
    n = max(10, int(per_km2 * area_km2))
    pts = cv2.goodFeaturesToTrack(
        cv2.GaussianBlur(g, (0, 0), 1.5 / gsd if gsd < 1.5 else 1.0),
        maxCorners=n, qualityLevel=0.01, minDistance=max(1.0, min_dist_m / gsd), blockSize=5,
    )
    return np.zeros((0, 2)) if pts is None else pts.reshape(-1, 2).astype(np.float64)


# --- line-segment junctions -----------------------------------------------------------------
def lsd_junctions(
    img: np.ndarray,
    gsd: float,
    min_len_m: float = 25.0,
    min_angle_deg: float = 35.0,
    reach_m: float = 6.0,
    merge_m: float = 12.0,
) -> np.ndarray:
    """Points where two long straight segments meet or cross (within `reach_m` of both)."""
    g = _clahe(_gray(img))
    g = cv2.GaussianBlur(g, (0, 0), max(0.6, 1.0 / gsd))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(g)[0]
    if lines is None:
        return np.zeros((0, 2))
    seg = lines.reshape(-1, 4).astype(np.float64)
    length = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    seg = seg[length >= min_len_m / gsd]
    if len(seg) < 2:
        return np.zeros((0, 2))
    p, q = seg[:, :2], seg[:, 2:]
    d = q - p
    L = np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    reach = reach_m / gsd
    # Candidate pairs: segments whose bounding boxes (grown by reach) overlap.
    from scipy.spatial import cKDTree

    mid = (p + q) / 2
    tree = cKDTree(mid)
    # Two segments can touch when their midpoints are up to (Li + Lj) / 2 + reach apart.
    pairs = tree.query_pairs(r=float(L.max()) + reach, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((0, 2))
    i, j = pairs[:, 0], pairs[:, 1]
    keep = (np.linalg.norm(mid[i] - mid[j], axis=1) <= (L[i] + L[j]) / 2 + reach)
    i, j = i[keep], j[keep]
    cross = u[i, 0] * u[j, 1] - u[i, 1] * u[j, 0]
    sin_min = math.sin(math.radians(min_angle_deg))
    keep = np.abs(cross) >= sin_min
    i, j, cross = i[keep], j[keep], cross[keep]
    # Intersection of infinite lines: p_i + t*u_i = p_j + s*u_j
    w = p[j] - p[i]
    t = (w[:, 0] * u[j, 1] - w[:, 1] * u[j, 0]) / cross
    s = (w[:, 0] * u[i, 1] - w[:, 1] * u[i, 0]) / cross
    ok = (t >= -reach) & (t <= L[i] + reach) & (s >= -reach) & (s <= L[j] + reach)
    x = p[i[ok]] + t[ok, None] * u[i[ok]]
    h, w_ = g.shape
    inside = (x[:, 0] >= 0) & (x[:, 0] < w_) & (x[:, 1] >= 0) & (x[:, 1] < h)
    # The edges of two crossing roads meet at up to 4 corners: merge them into the crossing center.
    return cluster_points(x[inside], merge_m / gsd)


# --- classic road skeleton junctions --------------------------------------------------------
def road_mask(img: np.ndarray, gsd: float, width_m: tuple[float, float] = (3.0, 15.0)) -> np.ndarray:
    """Rough road mask: thin structures brighter or darker than surroundings, low saturation."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    g = _clahe(_gray(img)).astype(np.float32)
    k = max(3, int(round(width_m[1] / gsd)) | 1)
    # White and black top-hat pick out thin bright / dark bands narrower than k.
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bright = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    lowsat = (hsv[..., 1] < 70).astype(np.float32)
    resp = bright * (0.5 + 0.5 * lowsat)
    thr = max(12.0, float(np.percentile(resp, 88)))
    mask = (resp > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # Drop blobs that are not line-shaped: keep components that are long relative to their area.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_len = 30.0 / gsd
    keep = np.zeros(n, bool)
    for c in range(1, n):
        x, y, w, h, area = stats[c]
        span = math.hypot(w, h)
        if span >= min_len and area / max(span, 1) <= width_m[1] / gsd * 1.5:
            keep[c] = True
    return keep[lab].astype(np.uint8)


def road_junctions(img: np.ndarray, gsd: float) -> np.ndarray:
    return junctions_from_mask(road_mask(img, gsd), gsd)


def junctions_from_mask(mask: np.ndarray, gsd: float, merge_m: float = 10.0, min_branch_m: float = 12.0) -> np.ndarray:
    """Skeletonize a road mask and return nodes where 3 or more branches meet."""
    from skimage.morphology import skeletonize

    if mask.sum() == 0:
        return np.zeros((0, 2))
    sk = skeletonize(mask.astype(bool)).astype(np.uint8)
    # Prune short spurs a few times so noise does not create fake junctions.
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(max(1, int(min_branch_m / gsd))):
        nb = cv2.filter2D(sk, -1, kernel) - sk
        ends = (sk == 1) & (nb == 1)
        if not ends.any():
            break
        sk[ends] = 0
    nb = cv2.filter2D(sk, -1, kernel) - sk
    ys, xs = np.nonzero((sk == 1) & (nb >= 3))
    pts = np.column_stack([xs, ys]).astype(np.float64)
    return nms_points(pts, merge_m / gsd)


# --- learned road segmentation ----------------------------------------------------------------
SEG_MODEL_ID = "kishan4444/upernet-swin-openearthmap-road-segmentation"  # UperNet + Swin-T, OpenEarthMap
SEG_GSD = 0.5  # OpenEarthMap imagery is 0.25-0.5 m/px
_seg = None


def _seg_model():
    global _seg
    if _seg is None:
        import torch
        from transformers import UperNetForSemanticSegmentation

        torch.set_num_threads(4)
        model = UperNetForSemanticSegmentation.from_pretrained(SEG_MODEL_ID).eval()
        labels = {v: int(k) for k, v in model.config.id2label.items()}
        _seg = (model, labels)
    return _seg


def segment(img: np.ndarray, gsd: float, tile: int = 512, overlap: int = 64) -> tuple[np.ndarray, dict]:
    """Per-pixel class ids at the input resolution, from tiled inference at SEG_GSD."""
    import torch

    model, labels = _seg_model()
    k = gsd / SEG_GSD
    x = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    h, w = x.shape[:2]
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    norm = ((x - mean) / std).astype(np.float32)
    n_cls = len(model.config.id2label)
    logits = np.zeros((n_cls, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    stride = tile - overlap
    ys = list(range(0, max(1, h - tile + stride), stride)) or [0]
    xs = list(range(0, max(1, w - tile + stride), stride)) or [0]
    with torch.inference_mode():
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                y0_, x0_ = max(0, y1 - tile), max(0, x1 - tile)
                patch = norm[y0_:y1, x0_:x1]
                ph, pw = patch.shape[:2]
                pad = np.zeros((tile, tile, 3), np.float32)
                pad[:ph, :pw] = patch
                out = model(pixel_values=torch.from_numpy(pad.transpose(2, 0, 1))[None]).logits[0].numpy()
                logits[:, y0_:y1, x0_:x1] += out[:, :ph, :pw]
                weight[y0_:y1, x0_:x1] += 1
    cls = (logits / np.maximum(weight, 1)).argmax(0).astype(np.uint8)
    cls = cv2.resize(cls, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cls, labels


def seg_road_junctions(img: np.ndarray, gsd: float) -> np.ndarray:
    cls, labels = segment(img, gsd)
    mask = (cls == labels["road"]).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] * gsd * gsd >= 300  # drop specks under 300 m2
    return junctions_from_mask(big[lab].astype(np.uint8), gsd)


EXTRACTORS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "corners": corners,
    "lsd": lsd_junctions,
    "roads": road_junctions,
    "segroads": seg_road_junctions,
}
