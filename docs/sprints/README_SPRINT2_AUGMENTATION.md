# Sprint 2: Augmentation Benchmarking

## Overview

This task benchmarks selected audio augmentation techniques identified in Sprint 1. The aim is to evaluate how each method affects CNN model performance.

## Methods Tested

- Baseline (no augmentation)
- Time Stretching
- Pitch Shifting
- Gaussian Noise Injection
- SpecAugment
- Combined Noise + SpecAugment

## Approach

A controlled experiment was completed using:

- A fixed subset of the dataset with 5 categories
- Mel spectrogram feature extraction
- A CNN model with the same architecture for all methods
- The same training settings for each benchmark run

The dataset was split into training, validation, and test sets. Augmentation was applied only to the training data, while validation and test data remained unchanged.

## Evaluation Metrics

- Accuracy
- Weighted F1 Score

## Results Summary

| Method | Accuracy | F1 Score | Decision |
|---|---:|---:|---|
| Noise Injection | 0.393 | 0.328 | Keep |
| Baseline | 0.286 | 0.167 | Reference only |
| Pitch Shift | 0.286 | 0.221 | Defer |
| Noise + SpecAugment | 0.214 | 0.129 | Defer |
| SpecAugment | 0.179 | 0.077 | Drop |
| Time Stretch | 0.143 | 0.145 | Drop |

## Key Findings

- Noise injection showed the best performance in this benchmark.
- Pitch shifting produced the same accuracy as the baseline, with a slight improvement in F1 score.
- SpecAugment, time stretching, and the combined Noise + SpecAugment method performed below the baseline.

## Final Recommendation

- Keep: Noise Injection
- Defer: Pitch Shift, Noise + SpecAugment
- Drop: SpecAugment, Time Stretch
- Reference only: Baseline

## Files Included

- `Sprint2_Augmentation_KD.ipynb` – main benchmarking notebook