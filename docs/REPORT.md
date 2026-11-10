# Benchmark report

Test set: 222 images (56 defective) from MVTec AD bottle, metal_nut, screw; split 60/20/20 stratified by product × defect type (seed 42). One model for all three products.

Operating threshold per model: highest value that keeps recall ≥ 95% on **validation**, then applied to test.

| Model | AUROC | AP | Recall | Precision | F1 | **FNR (missed defects)** | FPR (false alarms) | p50 latency | p95 |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 (ImageNet transfer) | 0.988 | 0.968 | 0.982 | 0.753 | 0.853 | **1.8%** (1) | 10.8% (18) | 154 ms | 185 ms |
| ResNet18 (ImageNet transfer) | 0.971 | 0.940 | 0.964 | 0.574 | 0.720 | **3.6%** (2) | 24.1% (40) | 137 ms | 172 ms |
| Small CNN (from scratch) | 0.741 | 0.603 | 0.982 | 0.301 | 0.460 | **1.8%** (1) | 77.1% (128) | 82 ms | 93 ms |

Same models at a naive 0.5 threshold, for contrast:

| Model | Recall | Precision | FNR | FPR |
|---|---|---|---|---|
| Small CNN (from scratch) | 0.518 | 0.354 | 48.2% | 31.9% |
| ResNet18 (ImageNet transfer) | 0.911 | 0.785 | 8.9% | 8.4% |
| EfficientNet-B0 (ImageNet transfer) | 0.911 | 0.864 | 8.9% | 4.8% |

## Error analysis: EfficientNet-B0 (ImageNet transfer) at its operating threshold

| Product | Images | Defects | Recall | FNR | FPR |
|---|---|---|---|---|---|
| bottle | 58 | 12 | 1.000 | 0.0% | 0.0% |
| metal_nut | 67 | 19 | 1.000 | 0.0% | 0.0% |
| screw | 97 | 25 | 0.960 | 4.0% | 25.0% |

Recall by defect type:

| Product / defect | Recall |
|---|---|
| screw/manipulated_front | 0.80 |
| bottle/broken_large | 1.00 |
| bottle/broken_small | 1.00 |
| bottle/contamination | 1.00 |
| metal_nut/bent | 1.00 |
| metal_nut/color | 1.00 |
| metal_nut/flip | 1.00 |
| metal_nut/scratch | 1.00 |
| screw/scratch_head | 1.00 |
| screw/scratch_neck | 1.00 |
| screw/thread_side | 1.00 |
| screw/thread_top | 1.00 |

Missed defects (1): `screw/manipulated_front/012.png`
