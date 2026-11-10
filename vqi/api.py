"""Inspection service: image -> defect decision (+ boxes), every inspection stored for history and stats.

    uvicorn vqi.api:app --port 8002
CLASSIFIER = resnet18 | efficientnet_b0 | cnn (default: best AP in artifacts/results.json)
DATABASE_URL defaults to SQLite in artifacts/; set a PostgreSQL URL in production.
"""
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import create_engine, text

from vqi.classify import EVAL_TF, load
from vqi.data import PRODUCTS

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
STORE = Path(os.getenv("INSPECTION_DIR", ROOT / "data" / "inspections"))
engine = create_engine(os.getenv("DATABASE_URL", f"sqlite:///{(ART / 'inspections.db').as_posix()}"))
MAX_BYTES = 10 * 2**20
M: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    results = json.loads((ART / "results.json").read_text())
    name = os.getenv("CLASSIFIER") or max((m for m in results if m != "yolo"), key=lambda m: results[m]["AP"])
    M["name"], M["net"] = name, load(name)
    th = json.loads((ART / "thresholds.json").read_text())
    M["tau"] = th[name]
    yolo_w = ART / "yolo" / "defect" / "weights" / "best.pt"
    if yolo_w.exists():
        from ultralytics import YOLO
        M["yolo"], M["tau_yolo"] = YOLO(str(yolo_w)), th.get("yolo", 0.25)
    STORE.mkdir(parents=True, exist_ok=True)
    with engine.begin() as c:
        c.execute(text("CREATE TABLE IF NOT EXISTS inspections (id VARCHAR(36) PRIMARY KEY, created_at VARCHAR(32), "
                       "product VARCHAR(32), filename VARCHAR(255), score FLOAT, decision VARCHAR(16), boxes TEXT, "
                       "model VARCHAR(32), latency_ms FLOAT)"))
    yield


app = FastAPI(title="Visual quality inspection", lifespan=lifespan)


@app.post("/inspect")
async def inspect(file: UploadFile = File(...), product: str = Form("unknown")):
    if product not in PRODUCTS + ["unknown"]:
        raise HTTPException(422, f"product must be one of {PRODUCTS}")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 10 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # rejects truncated or non-image payloads before decoding
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "not a decodable image")

    t0 = time.perf_counter()
    with torch.no_grad():
        score = float(torch.softmax(M["net"](EVAL_TF(img)[None]), 1)[0, 1])
    boxes = []
    if "yolo" in M:
        r = M["yolo"].predict(np.array(img)[:, :, ::-1], imgsz=640, conf=M["tau_yolo"], verbose=False, device="cpu")[0]
        boxes = [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": round(c, 3)}
                 for b, c in zip(r.boxes.xyxy.numpy().round(1).tolist(), r.boxes.conf.numpy().tolist())]
    ms = (time.perf_counter() - t0) * 1000
    decision = "defect" if score >= M["tau"] else "pass"

    rid = str(uuid.uuid4())
    img.save(STORE / f"{rid}.jpg", quality=90)  # kept for audit and re-labelling of disputed calls
    with engine.begin() as c:
        c.execute(text("INSERT INTO inspections VALUES (:id, :ts, :p, :f, :s, :d, :b, :m, :l)"),
                  {"id": rid, "ts": datetime.now(timezone.utc).isoformat(), "p": product, "f": (file.filename or "")[:255],
                   "s": score, "d": decision, "b": json.dumps(boxes), "m": M["name"], "l": ms})
    return {"id": rid, "decision": decision, "defect_score": round(score, 4), "threshold": round(M["tau"], 4),
            "boxes": boxes, "image_size": img.size, "model": M["name"], "latency_ms": round(ms, 1)}


@app.get("/history")
def history(limit: int = 50):
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM inspections ORDER BY created_at DESC LIMIT :n"), {"n": min(limit, 500)}).mappings().all()
    return [dict(r) | {"boxes": json.loads(r["boxes"])} for r in rows]


@app.get("/stats")
def stats():
    with engine.connect() as c:
        rows = c.execute(text("SELECT product, COUNT(*) AS n, SUM(CASE WHEN decision='defect' THEN 1 ELSE 0 END) AS defects, "
                              "AVG(latency_ms) AS avg_ms FROM inspections GROUP BY product")).mappings().all()
    return [dict(r) | {"defect_rate": r["defects"] / r["n"]} for r in rows]


@app.get("/inspection/{rid}/image")
def image(rid: str):
    try:
        uuid.UUID(rid)  # only ids we generated; no path traversal
    except ValueError:
        raise HTTPException(404)
    p = STORE / f"{rid}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.get("/health")
def health():
    return {"status": "ok", "classifier": M["name"], "threshold": M["tau"], "detector": "yolo" in M}


@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "ui.html")
