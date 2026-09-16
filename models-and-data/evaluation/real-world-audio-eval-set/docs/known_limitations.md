# Known Limitations and Labelling Uncertainties

**Author:** Sriram Bharadwaz Miryalkar | Sprint 1 Task 3.14: Real-World Audio Evaluation-Set Preparation

## 1. Source-audio access constraint (read this first)

Project Echo's real production dataset (the merged existing dataset, iNaturalist exports, and
Atlas of Living Australia recordings referenced throughout the Sprint 1/2 plans) is **not
stored in the GitHub repository**. The repo uses DVC (`.dvc`, `.dvcignore` at the repo root)
and a gitignored `.data/` folder, and Sprint 2's plan confirms the audio dataset lives in
Google Cloud Storage (Maneesh's Sprint 2 task is specifically "Dataset Cloud Migration from
GCP"). This assistant session has no GCP credentials and no access to DataBytes' DVC remote,
so **it was not possible to pull the project's actual field/real-world dataset** from here.

What this document instead used is every piece of **real, already-committed wildlife audio
found directly in the public GitHub repository**
(`github.com/DataBytes-Organisation/Project-Echo`, commit `01ebbd1`), specifically:

- `models-and-data/samples/store_audio/` — six labelled species `.wav` recordings (used as the
  basis of this evaluation set)
- `models-and-data/test_fixtures/` — `decoded.wav`, `pig.wav`, and two `sample_uploads/*.mp3`
  files
- `src/production/engine/yamnet_dir/` — `cat-goat-dingo.wav`, `cat-ul-goat.wav` (generic
  multi-animal test clips used for the legacy YAMNet sound-event-detection path, not
  Project-Echo-labelled species audio)

**Action needed from Sriram:** once you have DVC/GCP access set up locally, re-run
`scripts/build_evaluation_set.py` against the real dataset/iNaturalist/ALA sources instead of
(or in addition to) `models-and-data/samples/store_audio/`, and re-generate the metadata and
compatibility report. The script is written so that only `SOURCE_SPECIES` (the dict of source
files) needs to change — the condition-derivation and preprocessing-verification logic is
otherwise dataset-agnostic and can run unmodified.

## 2. Only 6 distinct real species recordings were found

`models-and-data/samples/store_audio/` contains exactly six `.wav` files, verified by SHA-256
to be six genuinely different recordings (not duplicates), each 22.05 kHz mono, 10 seconds:

| File | Species (scientific) | In production model's 123-class list? |
|---|---|---|
| `Alauda Arvensis.wav` | *Alauda arvensis* (Eurasian Skylark) | Yes (index 9) |
| `Capra Hircus.wav` | *Capra Hircus* (Domestic Goat) | Yes (index 21) |
| `Cervus Unicolour.wav` | *Cervus Unicolour* (Sambar Deer) | Yes (index 24) |
| `Pachycephala Rufiventris.wav` | *Pachycephala Rufiventris* (Rufous Whistler) | **No** — closest class is *Pachycephala simplex* |
| `Strepera Graculina.wav` | *Strepera Graculina* (Pied Currawong) | **No** — closest class is *Strepera versicolor* |
| `Sus Scrofa.wav` | *Sus Scrofa* (Wild Boar / Feral Pig) | Yes (index 110) |

Two of six (33%) do not match any class the production model can output. This was **not**
assumed — it was checked programmatically against
`src/production/engine/models/efficientnetv2/class_mapping.json` (123 classes total). See
`docs/audio_condition_categories.md` for how this is used as a deliberate
out-of-distribution signal rather than discarded.

No metadata, provenance, recording date/location, or licence information is stored anywhere
in the repository alongside these six files (no accompanying `.json`/`.csv`/README next to
`store_audio/`). They should be treated as **internal project reference audio only**, not
externally licensed or attributable recordings, until someone on the team who added them
confirms their origin. This is flagged per-row in `evaluation_metadata.csv`
(`source_note` column).

## 3. Duplicate / placeholder test fixtures found (excluded from the evaluation set)

- `models-and-data/test_fixtures/decoded.wav` and `models-and-data/test_fixtures/pig.wav` are
  **byte-identical** (same SHA-256 hash) despite different filenames — one is a renamed copy
  of the other, not two independent recordings. Neither was used as evaluation-set source
  audio (they were used only as a preprocessing-compatibility smoke check, see the
  compatibility report).
- The two `sample_uploads/*.mp3` files are also byte-identical to each other and are 10.4 s of
  generic uploaded test audio (not labelled wildlife audio) — used only to confirm the
  production preprocessing path accepts a non-WAV (MP3) input, not included in the species
  evaluation set.
- `src/production/engine/yamnet_dir/cat-goat-dingo.wav` and `cat-ul-goat.wav` are stereo,
  16 kHz, ~20 s generic multi-animal clips built for testing the legacy YAMNet
  sound-event-detection path. They are not Project-Echo-species-labelled and were excluded
  from the labelled evaluation set for that reason, though they would be a reasonable
  additional overlapping/multi-species stress test if the team wants one later.

## 4. Condition variants are synthetic, not naturally field-recorded

The `background_noise`, `weak_signal`, and `overlapping_calls` folders were **derived** from
the six clean recordings using deterministic signal processing (documented in
`audio_condition_categories.md` and reproducible via a fixed random seed). They are a
reasonable stand-in for Sprint 1 given no genuinely noisy/faint/overlapping field recordings
were found in the repository, but they are not a substitute for real field conditions.
Dinal's Sprint 2 "Field Calibration Validation and Threshold Deployment" task explicitly
expects to "apply temperature scaling and thresholds to the real-world evaluation set (from
Sriram)" — Dinal and the team should be aware this set's noisy conditions are synthetic until
genuine field/noisy recordings replace them.

## 5. Pre-existing Engine configuration inconsistency (found, not introduced, by this work)

The Engine repository has **two different preprocessing configurations**:

- `models/efficientnetv2/preprocess_config.json` (used by the currently active
  `efficientnetv2_tflite` inference path): 32 kHz, 128 mel bands, fmin 20 Hz / fmax 14000 Hz.
- `echo_engine.json` (used by the older `combined_pipeline()` function): 48 kHz, 260 mel
  bands, fmin 20 Hz / fmax 13000 Hz.

This evaluation set was verified against the **active** `preprocess_config.json` path only
(the model actually invoked in production, per `ACTIVE_INFERENCE_MODEL` in
`echo_engine.json`), since that is what a real-world evaluation set needs to be compatible
with. All 21 files pass that path with the model's exact expected input tensor shape
`(1, 1, 128, 313)`. See `evaluation_set/preprocessing_compatibility_report.md` for full
results. This configuration mismatch is the same kind of gap Krish's Sprint 1 integration-gap
task is meant to surface — it is documented here for completeness and cross-team awareness,
not as something this task is responsible for fixing.

## 6. Species common names

Common names in `evaluation_metadata.csv` (e.g. "Eurasian Skylark", "Sambar Deer") are
standard, widely-published common names for the scientific/binomial names already present in
the repository's own filenames and class-mapping file — not independently sourced or
inferred labels.

## Summary of what Sprint 2 should do with this

1. Replace `SOURCE_SPECIES` in `scripts/build_evaluation_set.py` with real field/DVC/GCP
   audio once Sriram has that access, and re-run.
2. Confirm the origin/licence of the six `store_audio/` recordings with whoever on the team
   added them, or replace them.
3. Decide with Krish/Anand which of the two preprocessing configurations
   (`preprocess_config.json` vs `echo_engine.json`) is authoritative, since Sprint 2
   calibration/quantisation work (Hoang, Dinal) depends on a single answer.
4. Optionally add real (not synthetic) noisy and overlapping-call field recordings once
   available, keeping the synthetic ones as a controlled/reproducible baseline rather than
   discarding them.
