# ALA Recovery Findings

## Sprint 2 Task
ALA and iNaturalist Dataset Acquisition Recovery

## Legacy Acquisition Investigation

The previous ALA acquisition process used an HTML scraper targeting ALA BioCache occurrence search pages.

Current testing showed that the historical HTML search route returns HTTP 403, preventing the scraper from accessing the expected page structure.

The legacy scraper depended on HTML elements such as:
- `recordRow`
- `nextLink`
- occurrence detail page audio elements

Because these elements are not available through the blocked response, the old scraper is no longer a reliable recovery method.

## Current ALA JSON Service

The ALA JSON occurrence service is still accessible and returned HTTP 200.

The historical data resource `dr341` still exists and contains a large number of occurrence records, including machine observations. However, current queries returned no sound records for this resource using either:
- `soundIDsCount:[1 TO *]`
- `multimedia:"Sound"`

Therefore, the historical `dr341` resource can no longer be used to recover the sound data expected by the old scraper.

## Current Sound Availability

A global ALA sound search returned 78,676 indexed sound records.

A controlled inspection of the first 100 records showed:
- 93 records from `dr1411` — iNaturalist Australia
- 6 records from `dr23984` — The Glenelg-Hopkins Soundscape Project - Frog Calls
- 1 record from `dr19123` — NatureMapr

This indicates that ALA still contains sound data, but most accessible sound records sampled are sourced from iNaturalist Australia.

## Recovery Decision

ALA records sourced from iNaturalist Australia should not be recovered again through ALA because this would duplicate the separate iNaturalist recovery process.

The recovery process should instead target small, separate non-iNaturalist ALA sound resources where practical.

The Glenelg-Hopkins Soundscape Project - Frog Calls (`dr23984`) is a suitable candidate for a controlled ALA recovery test.

## Root Cause Summary

The primary blockers in the previous ALA acquisition process are:

1. The original acquisition relied on fragile HTML scraping.
2. The historical HTML search endpoint currently blocks automated requests with HTTP 403.
3. The old `dr341` resource no longer exposes sound media through the current ALA index.
4. Current ALA sound data is dominated by iNaturalist-sourced records, which creates a duplication risk.
5. A new recovery approach should use the ALA JSON service and explicitly exclude iNaturalist-derived resources.

## Controlled Recovery Result

Following investigation of the legacy acquisition process, a new controlled
recovery method was implemented using the current ALA JSON occurrence service
and original-media API.

The recovery targeted the non-iNaturalist resource:

- Resource UID: `dr23984`
- Resource: The Glenelg-Hopkins Soundscape Project - Frog Calls
- Available indexed sound records: 1,648

A controlled sample of five records was recovered successfully.

### Recovery Summary

- Records requested: 5
- Valid audio files recovered: 5
- Failed/skipped downloads: 0
- Unique occurrences: 5
- Unique sound IDs: 5
- Unique SHA-256 hashes: 5
- Species represented: 3
- Audio format: FLAC
- Recorded licence: CC-BY 4.0 (Int) for all five recovered records

Species represented in the controlled sample:

- Crinia signifera: 3
- Litoria ewingii: 1
- Limnodynastes dumerilii: 1

All downloaded files were independently identified as valid FLAC audio files.
SHA-256 hashes were stored in the recovery metadata to support duplicate
detection and data-integrity checking.

## ALA Recovery Outcome

The ALA acquisition problem was recoverable, but the historical scraping
approach should not be reused.

The original HTML-based scraper is fragile and its historical search route
currently returns HTTP 403. The historical `dr341` resource remains accessible
through the JSON service but currently returns no sound records under the
tested sound filters.

Current ALA sound searches also showed substantial overlap with iNaturalist
Australia. To avoid duplicate acquisition, the controlled recovery instead
used a separate non-iNaturalist ALA resource (`dr23984`).

The replacement approach uses structured ALA JSON services for occurrence
discovery and the ALA original-media endpoint for audio retrieval. Recovered
files and metadata remain isolated from the newly balanced Sprint 2 dataset.
