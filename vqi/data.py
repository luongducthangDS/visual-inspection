"""MVTec AD (bottle, metal_nut, screw) -> supervised splits for classification and YOLO detection.

MVTec ships defects only in its test folder (it is an unsupervised benchmark). For supervised models we pool
all images of a product and split 60/20/20, stratified by product × defect type, seed fixed.
Boxes for YOLO come from the pixel masks: one box per connected component.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mvtec"
SPLITS = ROOT / "data" / "splits.csv"
YOLO = ROOT / "data" / "yolo"
PRODUCTS = ["bottle", "metal_nut", "screw"]
YOLO_SIZE = 640  # images are 900-1024 px; resized once on export
MIN_BOX_PX = 4   # drop mask specks smaller than this (in exported pixels)


MIRROR = "https://huggingface.co/datasets/foersben/mvtec-ad"  # CC BY-NC-SA 4.0 mirror of MVTec AD


def download(workers: int = 16) -> None:
    import concurrent.futures as cf
    import json
    import urllib.request

    files = []
    for p in PRODUCTS:
        tree = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/main/{p}?recursive=true"))
        files += [x["path"] for x in tree if x["type"] == "file"]

    def get(path: str) -> None:
        dst = RAW / path
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{MIRROR}/resolve/main/{path}", dst)

    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(get, files))
    print(f"{len(files)} files in {RAW}")


def index() -> pd.DataFrame:
    rows = []
    for p in PRODUCTS:
        for img in sorted((RAW / p).glob("*/*/*.png")):
            split_dir, kind = img.parts[-3], img.parts[-2]
            if split_dir not in ("train", "test"):
                continue
            mask = RAW / p / "ground_truth" / kind / f"{img.stem}_mask.png"
            rows.append({"path": img.relative_to(ROOT).as_posix(), "product": p, "defect_type": kind,
                         "label": int(kind != "good"), "mask": mask.relative_to(ROOT).as_posix() if mask.exists() else None})
    return pd.DataFrame(rows)


def make_splits(seed: int = 42) -> pd.DataFrame:
    df = index()
    strata = df["product"] + "/" + df["defect_type"]
    trv, test = train_test_split(df, test_size=0.2, stratify=strata, random_state=seed)
    train, val = train_test_split(trv, test_size=0.25, stratify=strata[trv.index], random_state=seed)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df.to_csv(SPLITS, index=False)
    return df


def boxes_from_mask(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Normalised (cx, cy, w, h) for each connected component of a binary mask."""
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8), connectivity=8)
    out = []
    scale = YOLO_SIZE / max(h, w)
    for x, y, bw, bh, _ in stats[1:n]:
        if bw * scale < MIN_BOX_PX or bh * scale < MIN_BOX_PX:
            continue
        out.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def export_yolo(df: pd.DataFrame) -> Path:
    """One class ("defect"); good images are exported with empty label files as negatives."""
    if YOLO.exists():
        shutil.rmtree(YOLO)
    for r in df.itertuples():
        stem = f"{r.product}_{r.defect_type}_{Path(r.path).stem}_{Path(r.path).parts[-3]}"
        img = cv2.imread(str(ROOT / r.path))
        s = YOLO_SIZE / max(img.shape[:2])
        (YOLO / "images" / r.split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / r.split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(YOLO / "images" / r.split / f"{stem}.png"), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
        boxes = boxes_from_mask(cv2.imread(str(ROOT / r.mask), cv2.IMREAD_GRAYSCALE)) if isinstance(r.mask, str) else []
        (YOLO / "labels" / r.split / f"{stem}.txt").write_text("".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes))
    cfg = YOLO / "data.yaml"
    cfg.write_text(f"path: {YOLO.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: defect\n")
    return cfg


if __name__ == "__main__":
    download()
    d = make_splits()
    print(d.groupby(["product", "split"])["label"].agg(["count", "sum"]).unstack())
    print(export_yolo(d))
