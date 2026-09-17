# Audio-Condition Categories — Real-World Evaluation Subset

**Author:** Sriram Bharadwaz Miryalkar | **Workstream:** Dataset and Evaluation
**Sprint 1 Task 3.14 output** (also referenced by Dinal Jason Fernando's and Hoang Lam Vu's
Sprint 1/2 calibration tasks, per the cross-task dependency table: *"Dinal, Hoang, and
Sriram... Agree on the evaluation metadata format and audio-condition categories early."*)

## Categories defined

| Category | Folder | Files | How it was produced |
|---|---|---|---|
| **Clean audio** | `evaluation_set/clean/` | 6 | Unmodified source species recordings, resampled to 22.05 kHz mono PCM16 WAV. No noise or attenuation added. |
| **Background noise** | `evaluation_set/background_noise/` | 6 | Each clean recording mixed with synthetic pink (1/f) noise at a controlled **+10 dB SNR**. Pink noise was chosen over white noise because it better approximates the spectral tilt of natural environmental/ambient sound (wind, distant water, insect chorus) than flat-spectrum white noise. |
| **Weak signal** | `evaluation_set/weak_signal/` | 6 | Each clean recording attenuated by **-12 dB**, then mixed with the same synthetic pink noise at **0 dB SNR**, simulating a distant or faint call close to the noise floor. |
| **Overlapping calls** | `evaluation_set/overlapping_calls/` | 3 | Two different clean species recordings mixed together at **0 dB relative level** (equal loudness), truncated to the shorter clip's length. Ground truth for these files is a *pair* of species labels, not a single label. |

Total: **21 evaluation files** (6 + 6 + 6 + 3).

## Why these four categories

These match the categories required by the Sprint 1 task brief ("clean audio, background
noise, weak signals, and overlapping sounds") and are compatible with the field-condition
categories Dinal's calibration-validation framework also needs ("clean audio, environmental
noise, weak calls, overlapping calls, and out-of-distribution audio" — Dinal's category list
additionally includes out-of-distribution audio, addressed below).

## Out-of-distribution note

Two of the six source species — **Pachycephala Rufiventris** (Rufous Whistler) and
**Strepera Graculina** (Pied Currawong) — are **not present** in the production model's
123-class list (`models/efficientnetv2/class_mapping.json`). Their clean/noise/weak variants
therefore double as genuine out-of-distribution examples: the model is expected to produce a
low-confidence or "nearest relative" misclassification (e.g. towards *Pachycephala simplex* or
*Strepera versicolor*, which are in the class list) rather than a correct match. This is flagged
per-row in `evaluation_metadata.csv` via the `in_model_class_list` column rather than kept in a
separate folder, so Dinal's OOD analysis can filter on that column directly. See
`known_limitations.md` for details.

## Why synthetic condition derivation, not natural field recordings

No genuinely noisy, faint, or overlapping-call field recordings were found anywhere in the
Project Echo GitHub repository (see `known_limitations.md` for the full search). The only
real wildlife audio available in-repo is six clean, isolated, single-species recordings. To
still deliver background-noise, weak-signal and overlapping-call examples for Sprint 1,
this script derives them from those six real recordings using deterministic, documented
signal processing (fixed random seed = 42, so the output is exactly reproducible). This is a
standard technique for building controlled evaluation conditions in bioacoustics (the same
principle Kiernan's SpecAugment work and Dinal's calibration-validation framework use) — but
it is **not** a substitute for genuine field recordings. Sprint 2 should replace or supplement
these with real field/noisy recordings once Sriram has access to the project's actual DVC/GCP
audio store (see `known_limitations.md`).
