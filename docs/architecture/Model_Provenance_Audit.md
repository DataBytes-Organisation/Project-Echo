# Model Provenance Audit

**Author**: Nolan Nguyen (Engine team) — produced while working on the Sprint 1
"Reproducible Model-Training Pipeline Configuration" task, as a related but
separately-scoped follow-up. Not a required Sprint 1 deliverable.

**Status**: Findings only. No production files were changed as part of this
audit — see "Recommended next steps" for what would need team sign-off
before acting on any of this.

## Why this exists

`docs/architecture/Engine_Documentation.md`'s "Edge Cases & Warnings" section
already recommends locking the sample rate during training and documenting
model metadata, but no such record exists for the model currently shipped in
this repository. While reviewing how the "selected model" was previously
trained for a separate task, it became clear the question "which model is
actually in production, and how was it trained?" does not have a single,
consistent answer in this repo. This document lays out exactly what is and
isn't known, with file-level evidence, so the team can make an informed
decision rather than continuing to build on an unverified assumption.

## Summary

There are **three different model artifacts**, trained by **three different
pipelines**, and the artifact actually served in production is **not** the
one described by the architecture documentation.

| | Location | Format | Trained by |
|---|---|---|---|
| **A — "echo_model"** | `src/production/engine/models/echo_model/1/` | TensorFlow SavedModel | `src/production/engine/optimised_engine_pipeline.ipynb` (TF/Keras, EfficientNetV2-B0 via TF-Hub) |
| **B — "efficientnetv2" (TFLite)** | `src/production/engine/models/efficientnetv2/efficientnetv2_project_echo.tflite` | PyTorch → ONNX → TFLite | `.delete/archive/Prototypes/engine/torch_impl/Integrate_EfficientNetV2_Engine/project_echo_train_save_efficientnetv2.ipynb` |
| **C — torch_impl pipeline** | (no weights produced until this sprint's port work) | PyTorch | `src/prototypes/engine/augmentation/main.py` (ported this sprint) |

## Finding 1 — The Dockerfile deploys a different model than the docs describe

- `src/production/engine/Engine.Dockerfile` copies `echo_engine_iot.py` into
  the container as the runtime entrypoint — **not** `echo_engine.py`, which
  is the script `docs/architecture/Engine_Documentation.md` documents in
  detail.
- `src/production/engine/echo_engine.json` line 22 sets
  `"ACTIVE_INFERENCE_MODEL": "efficientnetv2_tflite"`.
- `echo_engine_iot.py` (around line 154) reads that flag and, when set,
  loads `models/efficientnetv2/efficientnetv2_project_echo.tflite`
  (Candidate B) directly, **instead of** calling TensorFlow Serving for
  `echo_model` (Candidate A).
- Meanwhile `src/production/engine/models.config` (the TF-Serving config)
  still only serves `echo_model` and `weather_model`, and
  `src/production/README.md` (step 4 of the setup instructions) still tells
  new contributors to produce a model via `optimised_engine_pipeline.ipynb`
  and place it at `models/echo_model/1/`.

**Net effect**: the setup instructions, the TF-Serving config, and the
architecture documentation all describe Candidate A as "the" model, but the
container that actually runs in production defaults to Candidate B. Anyone
onboarding from the docs alone would not know this.

## Finding 2 — Candidate A's committed weights have no traceable training run

- `git log --follow` on `src/production/engine/models/echo_model/1/` shows
  one commit that actually touches the current weights: `573e4e3a`
  ("fix(ci): commit echo_model SavedModel so model_server builds with it").
- That commit's own message states the weights were force-added from
  "upstream" to unblock a CI/Docker build, and explicitly notes that a
  different, locally-trained 624MB model **was not used**.
- Earlier commit history on the same path (`9e6422e8`, `8203c7ef`,
  `14121db5`, etc.) shows iterative test-training runs of
  `optimised_engine_pipeline.ipynb`, but none of them is linked to the
  weights actually committed at `echo_model/1/` today.
- **Conclusion**: the currently-shipped Candidate A weights cannot be tied
  to any specific, reproducible training run in this repository's history.

## Finding 3 — Even the intended training notebook (Candidate A) is not fully reproducible as written

Read directly from `src/production/engine/optimised_engine_pipeline.ipynb`:

- No global seed is set for TensorFlow, NumPy, or Python's `random` module.
  Only the initial file-listing shuffle is seeded
  (`dataset_utils.index_directory(..., seed=42)`); everything downstream —
  `train_ds.shuffle(...)` (no seed), the random 5-second audio window, the
  `audiomentations.Compose` augmentations, and a ±2° image rotation — is
  unseeded.
- The train/val/test split (`train_split=0.8, val_split=0.19`) leaves only
  ~1% of the data as a test set — in the notebook's own recorded run,
  `Found 1580 files belonging to 15 classes`, that is **16 files** held out
  for testing across 15 classes.
- `echo_engine.json` (the shipped inference config) sets
  `"AUDIO_WINDOW": 500`, but `echo_engine.py` (line ~101) hardcodes
  `self.config['AUDIO_WINDOW'] = None` immediately after loading that file,
  silently overriding it. Inference and training happen to still agree
  (both effectively use `None`), but only because of an undocumented
  in-code override that directly contradicts the checked-in config value —
  a landmine for anyone who edits `echo_engine.json` expecting it to take
  effect.

## Finding 4 — Candidate B's training record is thinner still

- Per `README_EfficientNetV2_Engine_Integration.md` and
  `_trained_models/training_metrics.json` in the archive: **3 epochs**,
  `efficientnetv2_rw_s`, batch size 16, lr 1e-4, 123 classes, recorded test
  accuracy 0.872 — no seed recorded, and only aggregate metrics survive
  (no dataset split details beyond sample counts).

## Recommended next steps (not actioned in this audit)

1. **Team decision needed**: which candidate (A or B) is actually meant to
   be "the" production model going forward? The Dockerfile and the docs
   currently disagree, and that should be resolved explicitly rather than
   left implicit.
2. If Candidate A is kept as the intended model, add global seeding
   (TF/NumPy/`random`) and a real test split to
   `optimised_engine_pipeline.ipynb` before the next training run, and
   commit a small model-card file (seed, dataset snapshot/species list,
   final epoch, accuracy) alongside any future `echo_model/1/` update, so
   the gap found in Finding 2 doesn't recur.
3. Reconcile `echo_engine.json`'s `AUDIO_WINDOW` value with what
   `echo_engine.py` actually does, so the checked-in config stops
   contradicting the runtime behaviour.
4. This repo's new PyTorch/Hydra pipeline (Candidate C, `src/prototypes/engine/augmentation/`)
   already fixes the seeding gap for its own training runs (see that
   folder's README) and is a reasonable long-term replacement candidate,
   but it has not yet been trained to convergence on real data — see the
   real-data training run referenced in that folder's docs for the current
   state of that effort.
