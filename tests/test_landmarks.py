import cv2
import numpy as np
import pytest
from scipy.spatial import cKDTree

from gpsloc.landmarks import junctions_from_mask, lsd_junctions, nms_points

GRID = [80, 200, 320]  # road centerlines, pixels at 1 m/px


def _grid(size=400, width=8):
    mask = np.zeros((size, size), np.uint8)
    for v in GRID:
        cv2.line(mask, (v, 20), (v, size - 20), 1, width)
        cv2.line(mask, (20, v), (size - 20, v), 1, width)
    return mask


def _truth():
    return np.array([(x, y) for x in GRID for y in GRID], float)


def _recall(found, truth, tol):
    if len(found) == 0:
        return 0.0
    return float(np.mean(cKDTree(found).query(truth)[0] <= tol))


def test_nms_keeps_strongest():
    pts = np.array([[0, 0], [1, 0], [10, 10]], float)
    out = nms_points(pts, 3.0, scores=np.array([1.0, 5.0, 2.0]))
    assert sorted(map(tuple, out)) == [(1.0, 0.0), (10.0, 10.0)]


def test_skeleton_junctions_find_grid_crossings():
    found = junctions_from_mask(_grid(), gsd=1.0)
    assert _recall(found, _truth(), tol=4.0) == 1.0
    assert len(found) <= 9 + 4  # no flood of spurious nodes


def test_lsd_junctions_on_rendered_grid():
    img = np.full((400, 400, 3), 90, np.uint8)
    img[_grid() == 1] = (200, 200, 200)
    found = lsd_junctions(img, gsd=1.0)
    assert _recall(found, _truth(), tol=3.0) == 1.0
    assert len(found) == 9  # one point per crossing, not one per road-edge corner


@pytest.mark.parametrize("angle", [0, 33, 90])
def test_junctions_follow_rotation(angle):
    mask = _grid()
    R = cv2.getRotationMatrix2D((200, 200), angle, 1.0)
    rot = cv2.warpAffine(mask, R, (400, 400), flags=cv2.INTER_NEAREST)
    truth = cv2.transform(_truth().reshape(-1, 1, 2), R).reshape(-1, 2)
    found = junctions_from_mask(rot, gsd=1.0)
    assert _recall(found, truth, tol=5.0) >= 8 / 9
