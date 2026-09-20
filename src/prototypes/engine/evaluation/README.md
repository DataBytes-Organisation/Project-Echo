# Held-Out Baseline Re-Evaluation

This folder contains the Engine Team Sprint 2 held-out-only baseline workflow
and the shared evaluation manifest used by Praveen and Hoang.

## Shared manifest contract

`heldout_manifest.csv` contains one row per evaluation file with exactly these
columns:

```text
filepath,species,label_id,split
```

- `filepath` is a POSIX path relative to the shared evaluation dataset root.
- `species` is the model class label.
- `label_id` is the zero-based index in the production class mapping.
- `split` is always `heldout`.

The manifest contains only the 737 deterministic `reconstructed_heldout` rows
from the Sprint 1 audit manifest. The 405 `unverified_backfill` rows are not
included.

Validated manifest SHA-256:
`3bb017abc94d414209772c16dfd2c37a2e440cf611e73aa97d83df743ba2177e`.

## Reproduce the shared manifest

From the repository root:

```bash
python src/prototypes/engine/evaluation/heldout_baseline.py build-manifest \
  --source-manifest /path/to/balanced_validation_manifest.csv \
  --output src/prototypes/engine/evaluation/heldout_manifest.csv \
  --detailed-output .data/heldout_baseline/heldout_evaluation_manifest.csv
```

The source manifest is a Sprint 1 supporting artifact and is intentionally not
stored in GitHub. The optional detailed output retains its audit columns for a
fresh inference run with the existing baseline evaluator.

## Recompute the held-out-only report

The verified Sprint 1 predictions can be filtered without rerunning the model:

```bash
python src/prototypes/engine/evaluation/heldout_baseline.py summarize \
  --predictions /path/to/per_file_predictions.csv \
  --manifest src/prototypes/engine/evaluation/heldout_manifest.csv \
  --output-dir .data/heldout_baseline/results
```

This writes held-out-only metrics, per-species metrics and the updated
poor-performing-species list. Generated outputs stay outside GitHub.

The verified held-out-only result is:

- 737 files, with no failed predictions;
- 95.52% top-1 accuracy;
- 98.10% top-3 accuracy;
- 98.51% top-5 accuracy; and
- 16 poor-performing species at the configured recall threshold of 0.90.

These figures are recomputed from the verified Sprint 1 per-file predictions;
they do not include the 405 backfill predictions.

For a fresh inference run, use the detailed held-out manifest with
`selected_model_baseline_evaluation/baseline_evaluation.py evaluate`, the fixed
audio dataset and the production model artifacts.

## Balanced dataset dependency

Once the Sprint 2 balanced dataset is delivered, run the same production model
and reporting threshold on its agreed evaluation manifest. Keep that result
separate from this reconstructed held-out baseline so both remain comparable.
