# Sprint 2 Finalisation — Real-World Evaluation Set

**Task 3.14 (Sprint 2):** Real-World Evaluation Set Finalisation
**Target grade:** Standard | **Workstream:** Dataset and Evaluation

## Required work, addressed

**1. "Finalise a labelled real-world/field audio evaluation set with metadata and
audio-condition categories."**

Finalised: `evaluation_set/` contains 21 labelled files across the four required condition
categories (clean, background noise, weak signal, overlapping calls), each with a full
metadata row in `evaluation_metadata.csv` / `.json` (species, condition, source, derivation
method, duration, sample rate, SHA-256, and per-file preprocessing result). Categories are
documented in `audio_condition_categories.md`.

This is a **finalised Sprint 1 baseline**, not a finalised *production* dataset — see the
important caveat below and in `known_limitations.md`.

**2. "Verify preprocessing compatibility with the production model."**

Verified. `evaluation_set/preprocessing_compatibility_report.md` shows **21/21 files pass**
through a faithful port of the actual production preprocessing function
(`EchoEngine.efficientnetv2_preprocess_audio_bytes`, the currently active
`efficientnetv2_tflite` inference path per `echo_engine.json`), each producing the exact
tensor shape the deployed TFLite model expects, `(1, 1, 128, 313)`, with no NaN/Inf values.

**3. "Share the finalised set with Dinal (field calibration) and Praveen (baseline
alignment)."**

This section is that handoff note.

---

## Handoff note for Dinal Jason Fernando (Field Calibration Validation and Threshold Deployment)

Dinal's Sprint 2 task is to "apply temperature scaling and thresholds to the real-world
evaluation set (from Sriram)" and "produce calibrated-versus-uncalibrated results on real
audio." What you're getting:

- 21 labelled files across your 4 required field-condition categories, plus a built-in
  out-of-distribution signal: 2 of the 6 source species (*Pachycephala Rufiventris*,
  *Strepera Graculina*) are **not** in the production model's 123-class list — filter
  `evaluation_metadata.csv` on `in_model_class_list == False` to get your OOD subset directly,
  rather than needing a separate OOD folder.
- Every file already passes the production preprocessing path, so you can go straight to
  running the model and computing ECE/Brier/reliability diagrams without re-validating
  input compatibility yourself.
- **Important:** the `background_noise` and `weak_signal` conditions are **synthetically
  derived** (calibrated pink noise mixed at documented SNRs), not naturally field-recorded.
  This is fine as a Sprint 1/2 baseline for building and testing your calibration workflow,
  but your calibrated-vs-uncalibrated results should be described as validated against a
  synthetic-condition proxy until real field-noise recordings are substituted in (see
  `known_limitations.md`, point 4).
- Coordinate with Hoang per the plan's cross-task dependency note: keep your field/threshold
  validation and Hoang's calibration/quantisation metrics on the *same* evaluation manifest
  so results stay comparable.

## Handoff note for Praveen Wannakuwatte (Held-Out Baseline Re-Evaluation and Shared Manifest)

Praveen's Sprint 2 task establishes the shared evaluation manifest reused by calibration and
quantisation work. What you're getting:

- `evaluation_metadata.csv` / `.json` uses stable `file_id` and `relative_path` columns —
  suitable to reference directly from your shared manifest rather than duplicating file
  paths.
- Every file's expected species label is in `species_scientific_name` /
  `model_class_index`; the two out-of-class species are flagged via `in_model_class_list`
  so they can be excluded from, or specifically included in, your held-out accuracy
  calculation depending on what you want to measure.
- This set is small (21 files, 9 distinct source species-pairings) — it is meant as a
  **Sprint 1/2 compatibility and calibration baseline**, not a replacement for your main
  held-out accuracy benchmark against the full validation subset. Treat it as a
  cross-check, not your primary metric source.

## What's still open for Sprint 3 / beyond

Everything in `known_limitations.md` point 6 applies here as well: this set should be
regenerated against the project's actual DVC/GCP-hosted field dataset once Sriram has that
access, using the same `scripts/build_evaluation_set.py` pipeline (only the `SOURCE_SPECIES`
input needs to change).
