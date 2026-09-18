# iNaturalist Recovery Findings

## Sprint 2 Task
ALA and iNaturalist Dataset Acquisition Recovery

## Initial Recovery Test

The existing iNaturalist acquisition script was tested using a controlled sample with:
- 21 target species
- sounds only
- research quality observations
- maximum 2 observations per species
- 0.5 second delay between requests
- test data stored outside the Git repository

## Results

- 20 of 21 target species resolved to an iNaturalist taxon ID.
- 34 sound records were successfully downloaded.
- Metadata was successfully written to metadata.csv.
- Downloaded test data size was approximately 19 MB.
- Audio formats included MP3, WAV, and M4A.

## Issues Identified

1. `Canis lupus dingo` could not be resolved and was skipped.
2. `Chrysococcyx minutillus` did not receive an exact match and was mapped to `Chalcites minutillus`.
3. `Rattus norvegicus` returned no sound observations under the current filters.
4. `Phasianus colchicus` returned no sound observations under the current filters.
5. Some downloaded observations have missing licence codes or are marked as all rights reserved.
6. Downloaded audio uses mixed file formats, so format normalisation may be required before later model use.

## Current Assessment

The previous iNaturalist acquisition process is recoverable and functional. The primary remaining work is taxonomy validation, licence filtering, metadata cleaning, file-format validation, and controlled recovery rather than a complete rewrite.
