# Selected-Model Baseline Evaluation

## Summary

- Successfully evaluated files: 1142 / 1142
- Reconstructed held-out files: 737
- Unverified backfill files: 405
- Reconstructed held-out top-1 accuracy: 0.9552
- Reconstructed held-out top-3 accuracy: 0.9810
- Reconstructed held-out top-5 accuracy: 0.9851
- Unverified backfill top-1 accuracy: 0.9457
- Overall accuracy: 0.9518
- Top-3 accuracy: 0.9755
- Top-5 accuracy: 0.9790
- Macro precision: 0.9540
- Macro recall: 0.9436
- Macro F1: 0.9460
- Average TFLite inference time: 47.32 ms
- Poor-performing evaluated species (recall < 0.90): 14
- Unevaluated species with no compliant files: 1

## Model interface

- Input shape: [1, 1, 128, 313]
- Input dtype: float32
- Output shape: [1, 123]
- Output dtype: float32
- Output interpretation: logits_softmax

## Lowest-performing species

| Species | Support | Precision | Recall | F1 | Top-3 | Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Acanthiza reguloides | 10 | 0.889 | 0.800 | 0.842 | 1.000 | 1.000 |
| jabwar | 10 | 0.889 | 0.800 | 0.842 | 0.900 | 0.900 |
| Coracina papuensis | 5 | 1.000 | 0.800 | 0.889 | 1.000 | 1.000 |
| Cormobates leucophaea | 10 | 1.000 | 0.800 | 0.889 | 0.800 | 0.800 |
| Entomyzon cyanotis | 5 | 1.000 | 0.800 | 0.889 | 0.800 | 0.800 |
| Falco berigora | 10 | 1.000 | 0.800 | 0.889 | 0.900 | 1.000 |
| Falcunculus frontatus | 10 | 1.000 | 0.800 | 0.889 | 0.800 | 0.800 |
| Fulica atra | 10 | 1.000 | 0.800 | 0.889 | 0.800 | 0.800 |
| Megapodius reinwardt | 10 | 1.000 | 0.800 | 0.889 | 0.800 | 0.800 |
| Microeca flavigaster | 10 | 1.000 | 0.800 | 0.889 | 1.000 | 1.000 |
| Neophema pulchella | 5 | 1.000 | 0.800 | 0.889 | 0.800 | 0.800 |
| Anhinga novaehollandiae | 6 | 1.000 | 0.833 | 0.909 | 0.833 | 0.833 |
| Eurostopodus argus | 7 | 1.000 | 0.857 | 0.923 | 0.857 | 0.857 |
| Falco cenchroides | 8 | 1.000 | 0.875 | 0.933 | 0.875 | 1.000 |

## Limitations

- The original training notebook recorded an 80/20 stratified split with random_state=42 but did not save the split manifest. Reconstructed held-out files are selected first. To meet the 10-per-species requirement where possible, shortages are filled from the remaining eligible pool and flagged as `unverified_backfill`; those rows may overlap training and must not be presented as an independent test set.
- Many legacy segmented objects use names such as `region_start-end` and do not preserve a source-recording identifier. Exact duplicates are excluded using GCS MD5 metadata, but independence between clips from the same original recording cannot always be verified.
- Metrics are calculated only from successfully decoded and inferred files; failures are listed in `per_file_predictions.csv`.

## Evidence files

- `metrics_summary.json`
- `per_file_predictions.csv`
- `per_species_metrics.csv`
- `confusion_matrix.csv` and `confusion_matrix.png`
- `poor_performing_species.csv`
- `unevaluated_species.csv`
- `metrics_by_selection_tier.csv`
- `reproducibility.json`
