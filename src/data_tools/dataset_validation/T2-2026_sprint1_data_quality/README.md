# Audio Dataset Quality Assurance (QA) Pipeline – Sprint 1

## Overview

This folder contains a reusable dataset quality assurance (QA) pipeline for validating bioacoustic audio datasets before machine learning development. The pipeline performs automated checks on the dataset structure, labels, audio files, and metadata, generating quality reports while preserving the original dataset.

The pipeline follows a non-destructive approach, meaning records are identified and reported if they contain issues, but are not automatically modified or deleted.

---

## Sprint 1 Objectives

The Sprint 1 implementation addresses the following requirements:

- Review the existing dataset structure.
- Create automated checks for unreadable audio, missing labels, and inconsistent paths.
- Identify unsupported formats and invalid recording durations.
- Record audio data such as sampling rates and channels.
- Produce reusable dataset quality reports.
- Avoid automatically deleting invalid records.

---

## What Was Implemented

The pipeline includes:

- Dataset discovery and manifest generation.
- Folder structure validation.
- Source and species extraction.
- Species label normalisation.
- Audio validation, including:
  - Readability checks
  - Duration validation
  - Sampling rate recording
  - Channel detection
- Unsupported format detection.
- Invalid record identification.
- Acoustic quality analysis as an extension from the Sprint 1 tasks.
- Generation of reusable CSV quality reports.
- Generation of a standalone Python script version of the pipeline (functionality to be verified).

---


## Dataset Structure

The pipeline expects the dataset to follow the structure:

```text
datasets/
    source/
        species/
            audio_file.wav
```

Example:

```text
datasets/
    gcp/
        eastern_rosella/
            recording_001.wav
```

---

## Running the Pipeline in Sprint 1 

1. Place the dataset folder (or dataset ZIP file) in the project directory.
2. Open the Jupyter notebook.
3. Install the required Python packages.
4. Run the notebook from top to bottom.
5. Generated reports will be saved to the `reports` directory.

---

## Generated Reports

The pipeline generates reusable CSV reports, including:

- `dataset_manifest.csv`
- `structure_report.csv`
- `species_counts.csv`
- `acoustic_complexity_report.csv`
- `invalid_records.csv`

---

## Future Work

The current pipeline was demonstrated on a sample subset of the dataset, with additional external audio files and invalid file formats included to test edge cases. Future work will expand the pipeline to process the complete dataset using the same validation and reporting workflow.

A standalone Python script has also been generated from the notebook. Future work includes testing and verifying its functionality to ensure it produces the same outputs as the notebook implementation.