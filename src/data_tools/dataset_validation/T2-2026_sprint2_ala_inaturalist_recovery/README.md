# Sprint 2 ALA and iNaturalist Dataset Acquisition Recovery

This folder contains the Sprint 2 recovery work for ALA and iNaturalist wildlife
audio.

## Structure

- `scripts/`
  Recovery and validation scripts.

- `metadata/`
  Clean metadata and recovery manifests.

- `reports/`
  Investigation results, validation summaries, and blocker/root-cause reports.

- `recovered_data/ala/`
  Local ALA recovered audio.

- `recovered_data/inaturalist/`
  Reserved for separately stored iNaturalist recovered audio.

## Storage Rule

Recovered audio is intentionally excluded from Git.

Only scripts, reports, metadata, manifests, and directory placeholders are
committed.

Recovered ALA and iNaturalist data must not be merged directly into the newly
balanced Sprint 2 dataset.

Before reuse, records should be checked for:

- valid licence,
- correct taxonomy,
- file availability,
- duplicate content,
- source provenance.

## ALA

The legacy HTML scraper should not be reused.

Current controlled recovery uses:

1. ALA JSON occurrence search
2. ALA original-media API
3. content-type validation
4. SHA-256 hashing
5. separate metadata output

## iNaturalist

The archived collector remains usable for controlled acquisition, but its output
must be validated before reuse.

The Sprint 2 validation checks:

- file existence,
- licence usability,
- taxonomy review flags,
- duplicate observation/media records,
- SHA-256 content duplicates.
