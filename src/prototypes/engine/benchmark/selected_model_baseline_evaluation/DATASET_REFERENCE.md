# Dataset Reference

## Cloud sources

- Google Cloud project: `sit-23t1-project-echo-25288b9`
- `gs://project_echo_bucket_3/`: segmented audio for the 118 scientific-name classes
- `gs://project_echo_birdclef/`: audio for `brant`, `jabwar`, `sheowl`, `spodov` and `wiltur`

The fixed file list is stored in
`evidence/balanced_validation_manifest.csv`. Each row records the GCS URI,
object generation, GCS checksums, local relative path and SHA-256 digest.

Manifest SHA-256:
`76cd6a19958ecbbac2bd257714dddbc58c7fb9abe2769d418913e4ace79544c9`

## Selection summary

- Target: up to 10 reliable files per model class
- Selected audio files: 1,142
- Represented classes: 122 of 123
- Reconstructed-held-out files: 737
- Unverified backfill files: 405
- Exact duplicate SHA-256 hashes: 0
- Unreadable selected files: 0

Only correctly labelled, non-augmented audio with a supported extension was
eligible. Exact duplicates were removed using GCS MD5 metadata. Twenty classes
have documented shortages in `evidence/dataset_shortages.csv`.
`Aidemosyne modesta` has no compliant file because every available object for
that class is an augmentation-generated variant.

## Data handling

The audio files and dataset ZIP are intentionally excluded from GitHub. Project
members with the required Google Cloud permissions can retrieve the exact
objects using the GCS URIs in the manifest. The packaged dataset is stored with
the Sprint 1 supporting files in Teams.

The original model-training notebook did not save its train/test manifest.
Rows marked `unverified_backfill` may overlap training data, and this limitation
must be included when reporting the baseline metrics.
