"""YOLOv8n defect detector (one class) trained on boxes derived from MVTec masks.

    python -m vqi.detect            # train + evaluate; CPU ~1-2 h, Colab T4 ~10 min
Image-level score = highest box confidence (0 if no box), so detection is comparable with the classifiers.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
YOLO_DIR = ROOT / "data" / "yolo"
WEIGHTS = ART / "yolo" / "defect" / "weights" / "best.pt"
DEVICE = os.getenv("YOLO_DEVICE", "cpu")  # "0" on a GPU box; evaluation latency is always measured on CPU


def iou(a, b) -> float:
    """Boxes as (x1, y1, x2, y2)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_boxes(label_file: Path, w: int, h: int) -> list[tuple]:
    out = []
    for line in label_file.read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def image_scores(model, split: str, conf: float = 0.01) -> tuple[pd.DataFrame, list]:
    """Per-image max confidence + raw boxes for IoU analysis."""
    rows, matches = [], []
    for img in sorted((YOLO_DIR / "images" / split).glob("*.png")):
        r = model.predict(str(img), conf=conf, imgsz=640, verbose=False, device="cpu")[0]
        boxes = r.boxes.xyxy.numpy().tolist()
        confs = r.boxes.conf.numpy().tolist()
        h, w = r.orig_shape
        gts = gt_boxes(YOLO_DIR / "labels" / split / f"{img.stem}.txt", w, h)
        rows.append({"image": img.stem, "label": int("_good_" not in img.stem),
                     "score": max(confs, default=0.0)})
        matches.append({"image": img.stem, "gts": gts, "pred": list(zip(boxes, confs))})
    return pd.DataFrame(rows), matches


def box_quality(matches: list, conf: float) -> dict:
    """Mean best IoU per ground-truth box and share of GT boxes found at IoU>=0.5, at an operating threshold."""
    best = [max((iou(g, b) for b, c in m["pred"] if c >= conf), default=0.0) for m in matches for g in m["gts"]]
    return {"gt_boxes": len(best), "mean_best_iou": float(np.mean(best)) if best else 0.0,
            "box_recall@iou0.5": float(np.mean([b >= 0.5 for b in best])) if best else 0.0}


def train(epochs: int = 60) -> None:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(data=str(YOLO_DIR / "data.yaml"), imgsz=640, epochs=epochs, batch=16, device=DEVICE, seed=42,
                patience=15, project=str(ART / "yolo"), name="defect", exist_ok=True, workers=2, plots=True,
                degrees=180, flipud=0.5, fliplr=0.5, mosaic=0.5, verbose=False)
    evaluate()


def evaluate() -> None:
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    m = model.val(data=str(YOLO_DIR / "data.yaml"), split="test", imgsz=640, device="cpu", verbose=False,
                  project=str(ART / "yolo"), name="test_eval", exist_ok=True)
    (ART / "preds").mkdir(parents=True, exist_ok=True)
    out = {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "precision": float(m.box.mp), "recall": float(m.box.mr)}
    for split in ("val", "test"):
        scores, matches = image_scores(model, split)
        scores.to_csv(ART / "preds" / f"yolo__{split}.csv", index=False)
        (ART / "preds" / f"yolo__{split}_boxes.json").write_text(json.dumps(matches))
    (ART / "yolo_metrics.json").write_text(json.dumps(out, indent=2))
    print(out)


if __name__ == "__main__":
    evaluate() if sys.argv[1:] == ["eval"] else train()
