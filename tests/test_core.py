import numpy as np

from vqi.data import boxes_from_mask
from vqi.detect import iou
from vqi.report import at, pick_threshold


def test_one_box_per_component_and_specks_dropped():
    m = np.zeros((1000, 1000), np.uint8)
    m[100:200, 300:500] = 255   # 200 x 100 px blob
    m[700:900, 700:720] = 255   # second blob
    m[10:12, 10:12] = 255       # 2 px speck -> < MIN_BOX_PX after resize to 640
    boxes = sorted(boxes_from_mask(m))
    assert len(boxes) == 2
    cx, cy, w, h = boxes[0]
    assert np.allclose([cx, cy, w, h], [0.4, 0.15, 0.2, 0.1])


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-9


def test_threshold_keeps_target_recall():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(400)].astype(int)
    s = np.r_[rng.uniform(0.3, 1, 100), rng.uniform(0, 0.6, 400)]
    t = pick_threshold(y, s, 0.95)
    r = at(y, s, t)
    assert r["recall"] >= 0.95 and r["missed"] <= 5
    assert abs(r["FNR"] - (1 - r["recall"])) < 1e-12
