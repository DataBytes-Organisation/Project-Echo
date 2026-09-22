# Sriram Bharadwaz Miryalkar — Engine Team, Dataset and Evaluation

Project Echo | Trimester 2, 2026 | DataBytes Organisation

This package contains the completed work for both of Sriram's assigned Sprint tasks:

- **Sprint 1, Task 3.14 — Real-World Audio Evaluation-Set Preparation**
- **Sprint 2, Task 3.14 — Real-World Evaluation Set Finalisation**

## What's in this package

```
README.md                          <- this file
scripts/
  build_evaluation_set.py          <- the reusable pipeline (run this to regenerate everything)
  preprocess_config.json           <- copy of the production model's preprocessing config
  class_mapping.json               <- copy of the production model's 123-class species mapping
  echo_engine_config.json          <- copy of the legacy engine config (for the compatibility report)
evaluation_set/
  clean/                           <- 6 files: unmodified real species recordings
  background_noise/                <- 6 files: +10 dB SNR pink-noise condition
  weak_signal/                     <- 6 files: -12 dB attenuated + 0 dB SNR condition
  overlapping_calls/               <- 3 files: two-species mixtures
  evaluation_metadata.csv          <- one row per file: species, condition, source, preprocessing result
  evaluation_metadata.json         <- same content, machine-readable
  preprocessing_compatibility_report.md  <- real pass/fail results against the production Engine
docs/
  audio_condition_categories.md    <- how/why each of the 4 categories was built
  known_limitations.md             <- REQUIRED READING — what this set is and isn't, and why
  sprint2_finalisation_and_handoff.md  <- Sprint 2 finalisation note + handoff to Dinal & Praveen
```

## How this was built

All source audio and Engine code came from the public repository
`github.com/DataBytes-Organisation/Project-Echo` (commit `01ebbd1` on `main`). The six real
wildlife/livestock species recordings already committed under
`models-and-data/samples/store_audio/` were used as the evaluation set's foundation. Every
file in `evaluation_set/` — including the derived noise/weak/overlap conditions — was run
through a faithful, line-for-line port of the **actual production preprocessing function**
(`EchoEngine.efficientnetv2_preprocess_audio_bytes` in `src/production/engine/echo_engine.py`,
the currently active inference path). All 21 files pass, producing the exact tensor shape the
deployed TFLite model expects: `(1, 1, 128, 313)`.

**Read `docs/known_limitations.md` before treating this as a final, ready-to-submit dataset.**
It explains, honestly, what could and could not be sourced given the environment this was
built in has no access to the project's DVC/GCP-hosted real dataset, and lists specific
follow-up actions for Sriram once he has that access.

## Mapping to the Sprint 1 task's required outputs

| Required Sprint 1 output | Delivered as |
|---|---|
| Organised real-world evaluation subset | `evaluation_set/clean/`, `background_noise/`, `weak_signal/`, `overlapping_calls/` |
| Evaluation metadata file | `evaluation_set/evaluation_metadata.csv` / `.json` |
| Preprocessing compatibility results | `evaluation_set/preprocessing_compatibility_report.md` (21/21 pass) |
| Audio-condition categories | `docs/audio_condition_categories.md` |
| Known limitations and labelling uncertainties | `docs/known_limitations.md` |

## Mapping to the Sprint 2 task's required work

See `docs/sprint2_finalisation_and_handoff.md` for the finalisation summary and the
handoff note for Dinal (field calibration) and Praveen (baseline alignment).

## To regenerate or extend this

```
pip install librosa==0.9.2 soundfile numpy scipy
python3 scripts/build_evaluation_set.py
```

The script's `SOURCE_SPECIES` dictionary is the one place to edit when real field/DVC/GCP
audio becomes available — everything else (condition derivation, metadata generation,
preprocessing verification, report generation) runs unchanged.
