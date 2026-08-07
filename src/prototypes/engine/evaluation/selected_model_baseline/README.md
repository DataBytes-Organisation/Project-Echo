# Selected-Model Baseline Evaluation

This folder implements the Engine Team Sprint 1 balanced baseline evaluation
for the selected 123-class EfficientNetV2 TFLite model.

## Outputs

The workflow produces:

- a fixed balanced validation manifest with up to 10 files per species;
- a reusable TFLite evaluation script;
- per-file predictions, confidence, top-five predictions and inference time;
- overall accuracy, top-3 and top-5 accuracy;
- macro precision, recall and F1;
- per-species performance;
- held-out-versus-backfill metrics;
- confusion-matrix CSV and PNG;
- a poor-performing-species report; and
- hashes and environment details needed to reproduce the result.

The audio dataset is local-only and must not be committed to Git. The manifest,
reports and scripts are the reviewable evidence.

## Important split limitation

The model-training notebook used a stratified 80/20 train/test split with
`random_state=42`, but it did not save the exact split manifest. This workflow
reconstructs that split from the current GCS inventories in deterministic
`class_index, object_name, bucket` order. Reconstructed held-out files are
selected first. Where that tier has fewer than 10 files, the workflow backfills
from the remaining eligible pool to satisfy the task's balanced-dataset rule.
Those rows are marked `unverified_backfill` and `possible_training_overlap=true`
in the manifest. The backfilled metrics must not be presented as a fully unseen
test result. This limitation is repeated in the generated report.

Exact duplicate objects are removed using GCS MD5 metadata. Some legacy files
are named only as `region_start-end`, so their original recording identifier is
unavailable; independence between segments of the same recording cannot always
be verified.

## Prerequisites

1. Python 3.10-3.13.
2. Install `requirements.txt` in a virtual environment.
3. Install and authenticate Google Cloud CLI with a Deakin account that has
   read access to project `sit-23t1-project-echo-25288b9`.
4. Use a local output directory outside Git for downloaded audio.

The script never uploads, changes or deletes GCS objects.

## Reproduction commands

From the repository root, set reusable paths for your environment. PowerShell
examples are shown below.

```powershell
$eval = "src/prototypes/engine/evaluation/selected_model_baseline/baseline_evaluation.py"
$config = "src/prototypes/engine/evaluation/selected_model_baseline/evaluation_config.json"
$run = ".data/selected_model_baseline"
$gcloud = "C:/path/to/google-cloud-sdk/bin/gcloud.cmd"
```

Inventory the configured source buckets:

```powershell
python $eval --config $config inventory --gcloud $gcloud --output "$run/inventory.csv"
```

Reconstruct the held-out split and rank candidates:

```powershell
python $eval --config $config select --inventory "$run/inventory.csv" --output-dir $run
```

Download only the primary and reserve candidates:

```powershell
python $eval download --selection-audit "$run/selection_audit.csv" --gcloud $gcloud --cache-dir "$run/candidate_cache"
```

Decode candidates and create the fixed balanced dataset:

```powershell
python $eval --config $config finalize --downloaded-candidates "$run/downloaded_candidates.csv" --dataset-dir "$run/balanced_dataset" --output-dir $run
```

Run TFLite inference and generate the reports:

```powershell
python $eval --config $config evaluate --manifest "$run/balanced_validation_manifest.csv" --dataset-dir "$run/balanced_dataset" --output-dir "$run/results"
```

Run the unit tests:

```powershell
python -m pytest src/tests/unit/engine/test_selected_model_baseline.py
```

## Comparing with Krish's main pipeline

Use `per_file_predictions.csv` as the fixed comparison input. Run the same
manifest through the main Engine pipeline and join the two outputs by
`sha256` or `relative_path`. Compare predicted index, predicted label,
confidence and preprocessing failures. Do not compare results from different
file lists or preprocessing settings.
