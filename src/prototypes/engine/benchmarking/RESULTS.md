# Calibration Results — Sprint 1

Initial calibration of the production **EfficientNetV2 TFLite** classifier,
produced by `calibration_quant_benchmark.py --efficientnet`.

## Results (2,166 clips, all 123 classes)

| variant                      | accuracy | macro-F1 | ECE   | Brier | size  | latency     |
| ---------------------------- | -------- | -------- | ----- | ----- | ----- | ----------- |
| `efficientnetv2_tflite_fp32` | 0.958    | 0.960    | 0.070 | 0.078 | 85 MB | 139 ms/clip |

Reliability diagram: [`results_efficientnet/efficientnet_reliability.png`](results_efficientnet/efficientnet_reliability.png).
Raw numbers: [`results_efficientnet/efficientnet_calibration_results.csv`](results_efficientnet/efficientnet_calibration_results.csv).

**Reading:** ECE ~0.07 -> the model is mildly over-confident (top-label
confidence runs a few points above accuracy)

## Caveats

- **In-distribution:** eval clips come from the same Otways dataset the model was
  trained on (no guaranteed held-out split), so accuracy/ECE are optimistic.
- **Single variant:** only the shipped fp32 `.tflite` exists, so this is a
  calibration **baseline**, not an fp32-vs-quantised comparison.

Held-out split + the quantisation sweep are covered in
[`SPRINT2_QUANTISATION_PLAN.md`](SPRINT2_QUANTISATION_PLAN.md).

## Reproduce

```bash
python calibration_quant_benchmark.py \
  --efficientnet <models/efficientnetv2 dir> \
  --data <otways dataset> --max-per-class 20 \
  --out-dir results_efficientnet
```
