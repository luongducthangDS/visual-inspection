"""Compare classifiers and the detector on the same test images; write docs/REPORT.md.

Operating point: a missed defect (false negative) costs far more than re-checking a good part, so the
threshold is the highest one that keeps recall >= TARGET_RECALL on validation.
"""
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
ART, PRED, DOCS = ROOT / "artifacts", ROOT / "artifacts" / "preds", ROOT / "docs"
TARGET_RECALL = 0.95
NAMES = {"cnn": "Small CNN (from scratch)", "resnet18": "ResNet18 (ImageNet transfer)",
         "efficientnet_b0": "EfficientNet-B0 (ImageNet transfer)", "yolo": "YOLOv8n detector (max box conf)"}


def pick_threshold(y: np.ndarray, s: np.ndarray, target: float = TARGET_RECALL) -> float:
    pos = np.sort(s[y == 1])
    k = int(np.floor((1 - target) * len(pos)))  # allow at most k misses
    return float(pos[k]) if len(pos) else 0.5


def at(y: np.ndarray, s: np.ndarray, t: float) -> dict:
    p = s >= t
    tp, fp, fn, tn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum(), (~p & (y == 0)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": t, "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "FNR": fn / (tp + fn) if tp + fn else 0.0, "FPR": fp / (fp + tn) if fp + tn else 0.0,
            "missed": int(fn), "false_alarms": int(fp)}


def frames() -> dict:
    """model -> {split: DataFrame(label, score, product, defect_type, path)}"""
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    out = {}
    for m in ("cnn", "resnet18", "efficientnet_b0"):
        if (PRED / f"{m}__test.npy").exists():
            out[m] = {s: splits[splits["split"] == s].reset_index(drop=True).assign(score=np.load(PRED / f"{m}__{s}.npy"))
                      for s in ("val", "test")}
    if (PRED / "yolo__test.csv").exists():
        out["yolo"] = {}
        for s in ("val", "test"):
            d = pd.read_csv(PRED / f"yolo__{s}.csv")
            prod = d["image"].str.extract(r"^(bottle|metal_nut|screw)_")[0]
            d["product"] = prod
            d["defect_type"] = [img[len(p) + 1:].rsplit("_", 2)[0] for img, p in zip(d["image"], prod)]
            d["path"] = d["image"]
            out["yolo"][s] = d
    return out


def latency(model: str, paths: list[str]) -> dict:
    from PIL import Image
    if model == "yolo":
        from ultralytics import YOLO
        y = YOLO(str(ART / "yolo" / "defect" / "weights" / "best.pt"))
        f = lambda p: y.predict(str(ROOT / p), imgsz=640, verbose=False, device="cpu")  # noqa: E731
    else:
        import torch
        from vqi.classify import EVAL_TF, load
        net = load(model)

        def f(p):
            with torch.no_grad():
                return net(EVAL_TF(Image.open(ROOT / p).convert("RGB"))[None])
    f(paths[0])
    ms = []
    for p in paths[:50]:
        t0 = time.perf_counter()
        f(p)
        ms.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95))}


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("visual-inspection")
    F = frames()
    splits = pd.read_csv(ROOT / "data" / "splits.csv")
    test_paths = splits[splits["split"] == "test"]["path"].tolist()
    res, thresholds = {}, {}
    for m, d in F.items():
        va, te = d["val"], d["test"]
        t = pick_threshold(va["label"].to_numpy(), va["score"].to_numpy())
        thresholds[m] = t
        y, s = te["label"].to_numpy(), te["score"].to_numpy()
        r = {"AUROC": roc_auc_score(y, s), "AP": average_precision_score(y, s), "op": at(y, s, t), "at_0.5": at(y, s, 0.5)}
        r["latency"] = latency(m, test_paths)
        r["per_product"] = {p: at(g["label"].to_numpy(), g["score"].to_numpy(), t) | {"n": len(g), "defects": int(g["label"].sum())}
                            for p, g in te.groupby("product")}
        r["per_defect_recall"] = {f"{p}/{k}": float((g["score"] >= t).mean())
                                  for (p, k), g in te[te["label"] == 1].groupby(["product", "defect_type"])}
        r["missed"] = te[(te["label"] == 1) & (te["score"] < t)]["path"].tolist()
        res[m] = r
        with mlflow.start_run(run_name=m):
            mlflow.log_metrics({"AUROC": r["AUROC"], "AP": r["AP"], **{f"op_{k}": float(v) for k, v in r["op"].items()},
                                **{f"latency_{k}": v for k, v in r["latency"].items()}})

    test = splits[splits["split"] == "test"]
    L = ["# Benchmark report", "",
         f"Test set: {len(test)} images ({int(test['label'].sum())} defective) from MVTec AD bottle, metal_nut, screw; "
         "split 60/20/20 stratified by product × defect type (seed 42). One model for all three products.", "",
         f"Operating threshold per model: highest value that keeps recall ≥ {TARGET_RECALL:.0%} on **validation**, then applied to test.", "",
         "| Model | AUROC | AP | Recall | Precision | F1 | **FNR (missed defects)** | FPR (false alarms) | p50 latency | p95 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, r in sorted(res.items(), key=lambda kv: -kv[1]["AP"]):
        o = r["op"]
        L.append(f"| {NAMES[m]} | {r['AUROC']:.3f} | {r['AP']:.3f} | {o['recall']:.3f} | {o['precision']:.3f} | {o['f1']:.3f} | "
                 f"**{o['FNR']:.1%}** ({o['missed']}) | {o['FPR']:.1%} ({o['false_alarms']}) | {r['latency']['p50_ms']:.0f} ms | {r['latency']['p95_ms']:.0f} ms |")
    L += ["", "Same models at a naive 0.5 threshold, for contrast:", "", "| Model | Recall | Precision | FNR | FPR |", "|---|---|---|---|---|"]
    for m, r in res.items():
        o = r["at_0.5"]
        L.append(f"| {NAMES[m]} | {o['recall']:.3f} | {o['precision']:.3f} | {o['FNR']:.1%} | {o['FPR']:.1%} |")

    if (ART / "yolo_metrics.json").exists():
        ym = json.loads((ART / "yolo_metrics.json").read_text())
        from vqi.detect import box_quality
        bq = box_quality(json.loads((PRED / "yolo__test_boxes.json").read_text()), thresholds["yolo"])
        res["yolo"]["detection"] = ym | bq
        L += ["", "## Localisation (YOLOv8n, test split)", "",
              "| mAP@50 | mAP@50:95 | Box precision | Box recall | GT boxes | Mean best IoU @op | GT boxes found (IoU≥0.5) @op |",
              "|---|---|---|---|---|---|---|",
              f"| {ym['mAP50']:.3f} | {ym['mAP50_95']:.3f} | {ym['precision']:.3f} | {ym['recall']:.3f} | {bq['gt_boxes']} | "
              f"{bq['mean_best_iou']:.3f} | {bq['box_recall@iou0.5']:.1%} |"]

    best = max(res, key=lambda m: res[m]["AP"])
    r = res[best]
    L += ["", f"## Error analysis: {NAMES[best]} at its operating threshold", "",
          "| Product | Images | Defects | Recall | FNR | FPR |", "|---|---|---|---|---|---|"]
    for p, o in r["per_product"].items():
        L.append(f"| {p} | {o['n']} | {o['defects']} | {o['recall']:.3f} | {o['FNR']:.1%} | {o['FPR']:.1%} |")
    L += ["", "Recall by defect type:", "", "| Product / defect | Recall |", "|---|---|"]
    for k, v in sorted(r["per_defect_recall"].items(), key=lambda kv: kv[1]):
        L.append(f"| {k} | {v:.2f} |")
    short = lambda p: "/".join(Path(p).parts[-4:-3] + Path(p).parts[-2:]) if "/" in p else p  # noqa: E731
    L += ["", f"Missed defects ({len(r['missed'])}): " + ", ".join(f"`{short(p)}`" for p in r["missed"])]

    (DOCS / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (ART / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (ART / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n".join(L))


if __name__ == "__main__":
    main()
