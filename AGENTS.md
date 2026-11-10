# visual-inspection: agent notes

Project 03 in the AI Engineer portfolio roadmap: defect detection on MVTec AD (bottle, metal_nut, screw).

## 2026-11-10: status

- Done: split (1,107 images, 275 defective, 60/20/20 stratified, seed 42), CNN from scratch (15 epochs), ResNet18 and EfficientNet-B0 (12 epochs each, CPU, 4 threads), report, API + UI, tests (5 pass), README.
- Test results (222 images, 56 defects, threshold = recall ≥ 95% on validation): EfficientNet-B0 AP 0.968, 1/56 missed, FPR 10.8%; ResNet18 2/56 missed, FPR 24.1%; CNN AP 0.603, FPR 77%. All errors are on screw.
- Not done: YOLOv8n (run `notebooks/colab_yolo.ipynb` with `YOLO_DEVICE=0`, then copy `artifacts/yolo/defect/weights`, `artifacts/yolo_metrics.json`, `artifacts/preds/yolo__*` back and rerun `python -m vqi.report`). The notebook clones the GitHub repo, so push first.

## Gotchas

- Use `.venv` in this folder.
- Decoding 1024 px PNGs dominated each epoch (275 s); `classify._load` caches 256 px images.
- When stopping a background training loop, stop the parent bash too, or the loop starts the next model (this happened once: two runs overlapped).
- Do not use random crops for augmentation: a crop can remove the defect and flip the label.
