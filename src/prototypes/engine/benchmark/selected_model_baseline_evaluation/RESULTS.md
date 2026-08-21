# Sprint 1 Baseline Results

Evaluation date: 7 August 2026 (Australia/Sydney)

## Dataset

- Model classes: 123
- Classes with compliant files: 122
- Fixed validation files: 1,142
- Reconstructed held-out files: 737
- Unverified backfill files: 405
- Unreadable files: 0
- Exact duplicate SHA-256 hashes: 0
- Manifest SHA-256: `76cd6a19958ecbbac2bd257714dddbc58c7fb9abe2769d418913e4ace79544c9`

Twenty species have fewer than the requested 10 compliant files. All 10 cloud
objects for `Aidemosyne modesta` are augmentation-generated variants, so that
class has no compliant validation file. See `evidence/dataset_shortages.csv`.

## Model metrics

| Metric | Result |
|---|---:|
| Successful inference | 1,142 / 1,142 |
| Overall top-1 accuracy | 95.18% |
| Top-3 accuracy | 97.55% |
| Top-5 accuracy | 97.90% |
| Macro precision | 95.40% |
| Macro recall | 94.36% |
| Macro F1 | 94.60% |
| Average TFLite inference time | 46.78 ms |

The reconstructed-held-out tier achieved 95.52% top-1, 98.10% top-3 and
98.51% top-5 accuracy. The unverified-backfill tier achieved 94.57% top-1.

Fourteen evaluated species have recall below the documented 0.90 review
threshold. The lowest observed recall is 0.80. Full details are in
`evidence/poor_performing_species.csv` and `evidence/per_species_metrics.csv`.

## Interpretation limitation

The training notebook used a stratified 80/20 split with `random_state=42` but
did not save its file manifest. The held-out tier is a deterministic
reconstruction, not proof of the original split. Backfills are explicitly
marked `possible_training_overlap=true`. These results are appropriate as a
selected-model baseline but must not be described as a fully independent test
set.
