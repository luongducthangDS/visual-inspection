"""Inspection API against real artifacts (classifiers + report first); throwaway SQLite and image store."""
import io
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.getenv("ARTIFACTS_DIR", ROOT / "artifacts"))
pytestmark = pytest.mark.skipif(not (ART / "results.json").exists(), reason="train classifiers + run vqi.report first")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    os.environ["DATABASE_URL"] = f"sqlite:///{(tmp / 'i.db').as_posix()}"
    os.environ["INSPECTION_DIR"] = str(tmp / "img")
    from fastapi.testclient import TestClient

    from vqi.api import app
    with TestClient(app) as c:
        yield c


def _img(label: int, product: str) -> Path:
    s = pd.read_csv(ROOT / "data" / "splits.csv")
    row = s[(s["split"] == "test") & (s["label"] == label) & (s["product"] == product)].iloc[0]
    return ROOT / row["path"]


def test_defect_flagged_good_passes_and_stats(client):
    bad = client.post("/inspect", files={"file": ("bad.png", _img(1, "bottle").read_bytes(), "image/png")},
                      data={"product": "bottle"}).json()
    good = client.post("/inspect", files={"file": ("ok.png", _img(0, "bottle").read_bytes(), "image/png")},
                       data={"product": "bottle"}).json()
    assert bad["decision"] == "defect" and good["decision"] == "pass"
    assert bad["defect_score"] > good["defect_score"]

    stats = {r["product"]: r for r in client.get("/stats").json()}
    assert stats["bottle"]["n"] == 2 and stats["bottle"]["defect_rate"] == 0.5
    hist = client.get("/history").json()
    assert {h["id"] for h in hist} == {bad["id"], good["id"]}
    assert client.get(f"/inspection/{bad['id']}/image").status_code == 200


def test_rejects_bad_input(client):
    assert client.post("/inspect", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 415
    assert client.post("/inspect", files={"file": ("x.png", io.BytesIO(b"\x89PNG"), "image/png")},
                       data={"product": "laptop"}).status_code == 422
    assert client.get("/inspection/..%2F..%2Fsecret/image").status_code == 404
    assert client.get("/inspection/not-a-uuid/image").status_code == 404
