# Sprint 2 Blocker and Root-Cause Summary

## Task

ALA and iNaturalist Dataset Acquisition Recovery

## ALA Findings

The legacy ALA acquisition process relied on HTML scraping of ALA BioCache
occurrence pages.

Current testing showed that the historical HTML route returns HTTP 403, so the
old scraper can no longer access the expected `recordRow`, `nextLink`, and
audio-page elements.

The historical resource `dr341` remains indexed in ALA, but current sound
queries returned no sound-bearing records for that resource.

ALA globally still contains sound media. A controlled sample showed that many
current ALA sound records are sourced from iNaturalist Australia, creating a
duplication risk if ALA and iNaturalist are recovered independently without
source filtering.

A separate non-iNaturalist resource was therefore selected:

- Resource UID: `dr23984`
- Resource: The Glenelg-Hopkins Soundscape Project - Frog Calls

A controlled recovery retrieved five valid FLAC audio files.

All five:
- downloaded successfully,
- were verified as FLAC audio,
- had unique occurrence IDs,
- had unique sound IDs,
- had unique SHA-256 hashes,
- recorded a CC-BY 4.0 licence.

### ALA Root Cause

The main issue was not that ALA no longer provides sound media. The problem was
that the previous implementation depended on a fragile HTML scraping workflow
and on assumptions tied to an older resource.

The replacement recovery method should use structured ALA JSON services and the
ALA original-media API.

## iNaturalist Findings

The archived iNaturalist collection script was tested using a controlled
recovery configuration.

The test acquired 34 sound records and corresponding metadata.

The collector remains operational, but several data-quality issues were found:

- licensing is inconsistent across downloaded records,
- some records are all-rights-reserved or have missing licence information,
- taxonomy fallback can select an unintended taxon,
- file formats are mixed,
- the original collector does not perform content hashing,
- the original collector is not fully recovery-safe when files already exist.

One known taxonomy issue occurred when the requested name
`Chrysococcyx minutillus` did not exact-match and the collector used
`Chalcites minutillus` through its fallback behaviour.

`Canis lupus dingo` could not be resolved during the controlled acquisition.

## iNaturalist Validation Result

The 34 acquired records were passed through a validation step.

Results:

- Total records: 34
- Files present locally: 34
- Records with usable licence: 4
- Records requiring taxonomy review: 2
- Records involved in content duplicates: 0
- Records currently usable for recovery: 4

SHA-256 hashing was added to support content-level duplicate detection.

Records were only marked usable when:
- the downloaded file existed,
- the licence met the controlled recovery rules,
- the record was not marked for taxonomy review,
- the content was not identified as duplicated.

## Recovery Decision

Recovered ALA and iNaturalist data must remain separate from the newly balanced
Sprint 2 dataset.

The recovered audio files themselves are kept outside Git. The repository
contains scripts, validation reports, clean metadata, manifests, and storage
instructions only.

ALA records sourced from iNaturalist should not be recovered a second time
through ALA unless explicit cross-source deduplication and provenance handling
are added.

## Final Outcome

The previous acquisition work was recoverable, but both sources required
changes before their outputs could be treated as reliable.

ALA required migration away from HTML scraping to supported structured APIs.

iNaturalist required post-acquisition validation for taxonomy, licensing,
duplicate content, and file availability.

The Sprint 2 recovery process now provides a reproducible workflow for
controlled acquisition, validation, clean metadata generation, and separate
storage.
