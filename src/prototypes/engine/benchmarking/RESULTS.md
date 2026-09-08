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

# Sprint 2 progress

## fp32 baseline + class-count audit (no labels needed)

Latency/size measurement and a model-vs-mapping class-count audit on the
shipped fp32 `.tflite`, using synthetic input (`--fp32-baseline`, needs no dataset
or source Keras model). Raw numbers:
[`results_efficientnet/efficientnet_fp32_baseline.csv`](results_efficientnet/efficientnet_fp32_baseline.csv).

| variant                      | input            | output   | size    | latency (ms/inference)              |
| ---------------------------- | ---------------- | -------- | ------- | ----------------------------------- |
| `efficientnetv2_tflite_fp32` | (1, 1, 128, 313) | (1, 123) | 85.2 MB | mean 135.8, median 133.2, p95 180.2 |

Reproduce the baseline:

```bash
python calibration_quant_benchmark.py --fp32-baseline \
  --efficientnet <models/efficientnetv2 dir> \
  --out-dir results_efficientnet
```

## Full quantisation sweep (fp32 / float16 / dynamic-range / int8)

The shipped model is only a compiled `.tflite`, so it cannot be re-quantised
directly. The source graph was recovered from the **archived ONNX export**
(`.delete/archive/.../Integrate_EfficientNetV2_Engine/_trained_models/efficientnetv2_project_echo.onnx`),
which is the exact graph behind the production `.tflite`: converting ONNX ->
TF SavedModel -> fp32 TFLite reproduces the shipped model to within a
**max logit diff of ~1e-5**. All variants below are therefore faithfully "the
production model, quantised". Variants generated with `quantise_efficientnetv2.py`
(TensorFlow converter, per-channel int8), scored with `benchmark_quant_variants.py`.

Eval: 1,717 clips across all 123 classes. Raw numbers:
[`results_efficientnet/quant_sweep_results.csv`](results_efficientnet/quant_sweep_results.csv).
Pareto view:
[`results_efficientnet/quant_sweep_pareto.png`](results_efficientnet/quant_sweep_pareto.png).

| variant                    | accuracy | macro-F1 | ECE   | Brier | size    | latency* |
| -------------------------- | -------- | -------- | ----- | ----- | ------- | -------- |
| `float32` (baseline)       | 0.958    | 0.959    | 0.073 | 0.080 | 85.2 MB | 65.8 ms  |
| `float16`                  | 0.958    | 0.959    | 0.073 | 0.080 | 42.7 MB | 68.9 ms  |
| `dynamic_range`            | 0.953    | 0.955    | 0.084 | 0.091 | 23.1 MB | 46.9 ms  |
| `int8` (full PTQ)          | 0.756    | 0.753    | 0.219 | 0.402 | 24.4 MB | 19.6 ms  |
| `full_int8` (int8 I/O)     | 0.757    | 0.756    | 0.220 | 0.400 | 24.4 MB | 19.6 ms  |

\* per-clip latency on host CPU (TF 2.21 / XNNPACK), indicative only — real
on-device latency is a separate edge-benchmarking task.

**Findings**

- **float16 is free:** identical accuracy and ECE to fp32 at half the size ->
  recommended default.
- **dynamic-range is the sweet spot:** 3.7x smaller (85 -> 23 MB) and ~29%
  faster for only -0.5 pt accuracy and a small ECE cost.
- **int8 PTQ is not viable here:** accuracy drops -20 pts (0.96 -> 0.76) and ECE
  triples (0.073 -> 0.22). Post-training int8 breaks this EfficientNetV2 —
  recovering it needs **quantisation-aware training** (separate Sprint 2 task).
- int8 also **destroys calibration** (ECE 0.22), tying into the temperature-scaling
  calibration work: quantisation is another axis affecting production confidence.

### Held-out split (Praveen's shared manifest)

Re-scored against the verified Sprint 2 held-out manifest
(`src/prototypes/engine/evaluation/heldout_manifest.csv`, the
`reconstructed_heldout` split with possible training overlap excluded). 687 of
the 737 held-out files were present locally (the 50 missing are the external
xeno-canto alias classes, e.g. `brant/XC*.ogg`); eval covers 116/123 classes.
Raw numbers:
[`results_efficientnet/quant_sweep_heldout_results.csv`](results_efficientnet/quant_sweep_heldout_results.csv).

| variant                    | accuracy | macro-F1 | ECE   | Brier | size    | latency* |
| -------------------------- | -------- | -------- | ----- | ----- | ------- | -------- |
| `float32` (baseline)       | 0.958    | 0.944    | 0.060 | 0.069 | 85.2 MB | 53.1 ms  |
| `float16`                  | 0.959    | 0.944    | 0.060 | 0.069 | 42.7 MB | 62.7 ms  |
| `dynamic_range`            | 0.956    | 0.940    | 0.072 | 0.079 | 23.1 MB | 26.6 ms  |
| `int8` (full PTQ)          | 0.795    | 0.746    | 0.218 | 0.360 | 24.4 MB | 18.2 ms  |
| `full_int8` (int8 I/O)     | 0.795    | 0.746    | 0.216 | 0.358 | 24.4 MB | 18.2 ms  |

**Held-out confirms the in-distribution findings.** Accuracy is essentially
unchanged from the in-distribution pool (fp32 0.958 vs 0.958), so the model
genuinely generalises — the earlier "optimistic, no held-out split" caveat is
retired. The quant ordering holds exactly: float16 free, dynamic-range the sweet
spot, int8 PTQ collapses (-16 pts accuracy, ECE tripled).

**Caveats:** latency is host-CPU indicative, not on-device. **Production runtime
compatibility:** the variants are built with TF 2.21, and the `dynamic_range`
variant emits a `FULLY_CONNECTED` op version the pinned **TF 2.10 engine runtime
cannot load** (`float16`/`int8`/`full_int8`/`fp32` load fine on TF 2.10). Shipping
`dynamic_range` needs either a runtime bump or a rebuild targeting an older op set.

Reproduce the sweep:

```bash
# 1. one-off, modern-TF env (NOT the pinned TF 2.10 venv):
onnx2tf -i efficientnetv2_project_echo.onnx -o effnet_savedmodel -nuo
# 2. generate variants + score them:
python quantise_efficientnetv2.py --saved-model effnet_savedmodel \
  --calib-npy calib_nhwc.npy --out-dir results_efficientnet/variants
# in-distribution (folder-per-class glob):
python benchmark_quant_variants.py --variants-dir results_efficientnet/variants \
  --model-dir ../../production/engine/models/efficientnetv2 \
  --data <otways dataset> --max-per-class 15 \
  --out-csv results_efficientnet/quant_sweep_results.csv
# held-out split (score exactly the shared manifest):
python benchmark_quant_variants.py --variants-dir results_efficientnet/variants \
  --model-dir ../../production/engine/models/efficientnetv2 \
  --data <otways dataset> \
  --manifest ../evaluation/heldout_manifest.csv \
  --out-csv results_efficientnet/quant_sweep_heldout_results.csv
```
