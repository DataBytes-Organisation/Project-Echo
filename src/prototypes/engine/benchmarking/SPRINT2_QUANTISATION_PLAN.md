# Sprint 2 — TFLite Quantisation Evaluation Plan

Engine team · follows the Sprint 1 _Calibration & TFLite Quantisation Evaluation Framework_
(`calibration_quant_benchmark.py`).

## 1. Where Sprint 1 landed

- Built a model-agnostic framework that computes **accuracy, macro-F1, ECE and
  Brier score**, draws reliability diagrams, and converts a Keras model to four
  TFLite variants (**fp32, dynamic-range int8, float16, full int8**), evaluating
  each on accuracy, calibration, on-disk size and latency.
- Validated end-to-end via `--selftest`.
- **Gap carried into Sprint 2:** the committed `echo_model` SavedModel outputs a
  single scalar (`[None, 1]`), so it cannot produce multi-class probabilities.
  The real classifier (EfficientNetV2, currently archived in `.delete/`) and its
  label ordering are needed for meaningful _initial_ numbers.

## 2. Sprint 2 objectives

### 2.1 Run on the real production model

- Recover the EfficientNetV2 Project-Echo classifier from `.delete/` (or retrain),
  export a clean Keras/SavedModel with a **known label order**, and confirm the
  audio -> input preprocessing (`img_size`, `n_mels`, `sr`) matches training.
- Produce the _real_ fp32-vs-quantised calibration comparison the framework is
  built for.

### 2.2 Full int8 with a proper representative dataset

- Replace the ad-hoc representative sampler with a **stratified sample of real
  audio** (all classes, ~100–500 clips) so activation ranges are well estimated.
- Quantify **accuracy retention** and **calibration drift** (ΔECE, ΔBrier)
  fp32 -> int8; investigate per-layer / mixed-precision fallback where int8 hurts.

### 2.3 Post-hoc calibration

- Add **temperature scaling** (fit T on a validation split) and report ECE/Brier
  before vs after, for both fp32 and int8 variants.
- Compare against isotonic / Platt scaling as stretch goals.

### 2.4 On-device latency

- Measure TFLite latency on the **actual IoT edge target** (Raspberry Pi via
  `src/production/iot/edge_inference/`), not just dev-CPU — report ms/sample and
  memory for each variant.

### 2.5 Trade-off analysis & recommendation

- Build the accuracy <-> ECE/Brier <-> size <-> latency Pareto view across variants and
  **recommend a deployment configuration** for the edge nodes.
- Add **per-class calibration** to surface species where quantisation degrades
  reliability the most.

## 3. Metrics & success criteria

| Metric                                  | Target                                  |
| --------------------------------------- | --------------------------------------- |
| Top-1 accuracy retention (int8 vs fp32) | ≤ 2% absolute drop                      |
| ECE after temperature scaling           | < 0.05                                  |
| Model size reduction (int8 vs fp32)     | ≥ 3× smaller                            |
| Edge latency (int8)                     | meets real-time budget on the Pi target |

## 4. Datasets

- Otways audio pulled from GCS (`project_echo_bucket_1/2/3`, `project_echo_birdclef`;
  124–125 species). Held-out split for eval + a separate representative/validation
  split for quantisation and temperature fitting.

## 5. Risks & mitigations

- **Label-order mismatch** between model and dataset -> require an explicit label
  file; the framework already accepts `--labels`.
- **Preprocessing mismatch** silently wrecks accuracy -> verify against training
  config before trusting numbers.
- **Full int8 conversion failures** on some ops -> fall back to dynamic-range /
  float16 and document unsupported ops.

## 6. Task breakdown (checklist)

- [ ] Recover/retrain EfficientNetV2 classifier -> clean Keras + label list
- [ ] Real representative & held-out eval splits from GCS audio
- [ ] Wire real model into `benchmark_variants` -> initial real ECE/Brier table
- [ ] Add temperature scaling + before/after calibration report
- [ ] On-device (Pi) latency harness
- [ ] Pareto trade-off report + deployment recommendation
