# Visual quality inspection: defect detection on product images

Camera or uploaded images of products are classified as pass or defect. The defect regions are
localised, every inspection is stored, and defect rates are tracked per product. The project compares
a CNN trained from scratch with ImageNet transfer learning (ResNet18, EfficientNet-B0), plus a YOLOv8n
detector, and sets the decision threshold by the cost that matters on a line: **missed defects**.

## Data

[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) (CC BY-NC-SA 4.0), three products:
bottle, metal_nut, screw. That is 1,107 images, 275 of them defective, across 12 defect types. Screw
defects are a few pixels wide, the realistic hard case.

MVTec is an unsupervised benchmark (defects appear only in its test folder). For supervised models,
all images of each product are pooled and split 60/20/20, stratified by product × defect type, seed
42 (`data/splits.csv`). YOLO boxes are derived from the pixel masks, one box per connected component.

## Results (test set: 222 images, 56 defective)

Threshold per model = the highest value that keeps recall ≥ 95% on **validation**, applied unchanged to test:

| Model | AUROC | AP | Missed defects (FNR) | False alarms (FPR) | Precision | p50 latency (CPU) |
|---|---|---|---|---|---|---|
| **EfficientNet-B0, ImageNet transfer** | **0.988** | **0.968** | **1 / 56 (1.8%)** | **10.8%** | 0.753 | 154 ms |
| ResNet18, ImageNet transfer | 0.971 | 0.940 | 2 / 56 (3.6%) | 24.1% | 0.574 | 137 ms |
| Small CNN from scratch (4 conv blocks) | 0.741 | 0.603 | 1 / 56 (1.8%) | 77.1% | 0.301 | 82 ms |

How to read it:
- **Transfer learning is the difference**: with roughly 400 training images, the from-scratch CNN
  reaches AP 0.60 against 0.97. Its low miss rate is bought by flagging 77% of good parts. A miss rate
  means nothing without the false-alarm rate next to it.
- **The threshold matters as much as the model.** At a naive 0.5, every model misses 8.9% of defects;
  EfficientNet's recall-targeted threshold cuts that to 1.8% at the price of more re-checks (FPR 4.8%
  → 10.8%).
- **Per product (EfficientNet):** bottle and metal_nut have 0 misses and 0 false alarms. All errors
  are on screw: 1 missed `manipulated_front` and a 25% false-alarm rate. Screw defects are tiny at
  256 px input, so the next step is higher resolution or tiling for that product.

Localisation (YOLOv8n, one "defect" class): training is set up but runs on a GPU
(`notebooks/colab_yolo.ipynb`, about 10 min on a T4; hours on CPU). When its weights are present,
`vqi.report` adds mAP@50, mAP@50:95 and IoU, and the API draws its boxes. Not yet run.

Full report: [docs/REPORT.md](docs/REPORT.md).

## Pipeline

```
MVTec images + masks ─► vqi/data.py      stratified split; masks → YOLO boxes; YOLO export (640 px)
                     ─► vqi/classify.py  CNN / ResNet18 / EfficientNet-B0, class-weighted loss,
                                          flips + rotation (no random crops: a crop can remove the defect)
                     ─► vqi/detect.py    YOLOv8n; image score = max box confidence
                     ─► vqi/report.py    recall-targeted thresholds, FNR/FPR, per product & defect type,
                                          latency, MLflow
                     ─► vqi/api.py       /inspect (upload or camera frame) → decision + boxes,
                                          stored in SQL with the image; /history, /stats; ui.html
```

## Run

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python -m vqi.data                                   # download ~500 MB, split, YOLO export
python -m vqi.classify cnn                           # also resnet18, efficientnet_b0
python -m vqi.detect                                 # optional; YOLO_DEVICE=0 on a GPU
python -m vqi.report
pytest -q
uvicorn vqi.api:app --port 8002                      # UI at http://localhost:8002 (upload or camera)
docker compose up --build                            # API + PostgreSQL, after training
```

## Limits

- MVTec photos are taken under controlled light, with one centred object and a clean background. A
  real line has motion blur, varying light, and defect types nobody labelled. These numbers say the
  approach works, not that it is ready for a specific factory.
- The test set is small (56 defects). One missed defect moves FNR by 1.8 points, so the gap between
  EfficientNet and ResNet18 is 1 image.
- A supervised classifier only knows the 12 defect types it has seen. A new defect type is the
  strongest argument for adding an anomaly-detection model trained on good parts only.
- Latency is single-image CPU inference on a laptop; a line camera needs a GPU or a smaller model to
  keep up with throughput.
