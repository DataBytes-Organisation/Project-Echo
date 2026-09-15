"""
Builds Sprint 2 Section 3.4's third Expected Output: "expanded augmented
training dataset (on the final balanced dataset)."

What "expanded" means here (confirmed with the project lead): targeted
oversampling of the lower-count species in Manisha/Raveesha's balanced
train/ split, using audio-domain augmentation to synthesise new training
clips - not a bigger crawl, not more raw recordings, and not the
SpecAugment masking Sprint 1 built (that stays exactly where it is,
applied on-the-fly per epoch by augment.py/light.yaml - see
individual_docs/augmentation_benchmark_results_and_selected_configuration.md
for why `light` was selected). This script produces a physical, on-disk
training set, separate from that per-epoch masking, that a team member
can point straight at.

Why this is needed on top of the balancing that already happened: Manisha
and Raveesha's floor=30/cap=150 balancing already guarantees every species
has at least 30 files and at most 150, which is a huge improvement over
the raw dataset - but it still leaves a real 5x spread between the
smallest and largest classes (verified directly: 79 species, counts
range 30-150, mean 79.2). That residual imbalance is exactly what this
script targets: TARGET_COUNT = 150 was chosen deliberately to reuse the
team's own already-agreed ceiling rather than invent a new threshold -
every species below it gets synthetically expanded up to it via
audio-domain augmentation; every species already at 150 is left
completely untouched (copied through as-is, nothing added or removed).

Why audio-domain augmentation (audiomentations), not SpecAugment: SpecAugment
(augment.py) operates on spectrograms, applied on-the-fly inside
SpectrogramDataset during training - it has no natural "save this as a new
audio file" step (masking a spectrogram and inverting back to a waveform
would introduce artifacts, not a real new recording). Audio-domain
transforms operate on the waveform itself, so their output is a normal,
valid audio file that can be written to disk and treated exactly like any
other training clip - including still getting SpecAugment applied to it
later, on-the-fly, during actual training, same as every other file. The
two are complementary, not duplicates of each other.

Why audiomentations specifically, and only self-contained transforms:
config.yaml's own augmentations.audio group already uses a waveform-level
augmentation library (torch_audiomentations) for AddBackgroundNoise - but
that needs an external background_noise/ folder this repo doesn't have,
which is exactly why every benchmark run this sprint stripped
~augmentations.audio, and torch_audiomentations isn't even installed in
the projectecho conda env this script (and every training run this
sprint) actually runs in. audiomentations (the CPU library, already
pinned in this project's dependencies) is used instead, restricted to
transforms that need no external assets: Gain, PitchShift, TimeStretch,
AddColorNoise (a synthetic-noise equivalent of the AddColoredNoise block
config.yaml itself has commented out, for the same missing-dependency
reason), and PolarityInversion. Nothing here removes or masks acoustic
content the way heavier distortions (BitCrush, RoomSimulator,
ClippingDistortion) would - the goal is new, still-recognisable examples
of the same species call, not a stress test.

Why this can't leak into validation/test: it only ever reads from and
writes into train/ - Manisha/Raveesha's validation/ and test/ folders are
never opened by this script. Augmented clips are synthesised from
train-only source files and land in train_expanded_v2/ only, so the
source-recording-level split the team built specifically to prevent
leakage stays intact.

Output layout: train_expanded_v2/<species>/... - a complete, self-contained
drop-in replacement for train/ (every original file is copied through
unchanged, plus the new augmented files), not an augmented-only
supplement someone has to manually merge with the original train/
folder. Point run_final_benchmark.py's +system.train_dir at it directly.
"""

import json
import random
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf
from audiomentations import AddColorNoise, Compose, Gain, PitchShift, PolarityInversion, TimeStretch

SEED = 42
TARGET_COUNT = 150  # Manisha/Raveesha's own balancing cap - see module docstring for why
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac")

# Four `.parent`s reach the repo root (Project-Echo/) from this file's
# folder (src/prototypes/engine/augmentation/) - same convention as
# FINAL_DATASET_DIR in run_augmentation_benchmark.py.
SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_DATASET_DIR = SCRIPT_DIR.parent.parent.parent.parent / "models-and-data" / "final_data_files"
SOURCE_DIR = FINAL_DATASET_DIR / "train"
# Named _v2 because the first build attempt (killed partway through by a
# multichannel-audio bug, now fixed above) left a partial train_expanded/
# that OneDrive held locked while syncing it - not a version of the
# augmentation approach, just this folder's actual name going forward.
OUTPUT_DIR = FINAL_DATASET_DIR / "train_expanded_v2"
MANIFEST_PATH = SCRIPT_DIR / "expanded_training_dataset_manifest.json"


def build_chain():
    """A fresh Compose instance per species (not shared/reused across the
    whole run) so each species' random draws don't share RNG state in a
    way that would make one species' augmentation choices depend on how
    many files an earlier species needed - each call still randomises
    independently either way, this just keeps the intent explicit."""
    return Compose([
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
        AddColorNoise(p=0.4),
        PolarityInversion(p=0.2),
    ])


def list_audio_files(species_dir):
    return sorted(f for f in species_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if OUTPUT_DIR.exists():
        raise SystemExit(f"{OUTPUT_DIR} already exists - delete it first if you want to rebuild.")
    OUTPUT_DIR.mkdir(parents=True)

    species_dirs = sorted(d for d in SOURCE_DIR.iterdir() if d.is_dir())
    manifest = {"seed": SEED, "target_count": TARGET_COUNT, "species": {}}

    for species_dir in species_dirs:
        species = species_dir.name
        src_files = list_audio_files(species_dir)
        original_count = len(src_files)

        dest_dir = OUTPUT_DIR / species
        dest_dir.mkdir(parents=True)

        # Copy every original file unchanged - see module docstring for
        # why train_expanded_v2/ has to be a complete replacement, not an
        # augmented-only supplement.
        for f in src_files:
            shutil.copy2(f, dest_dir / f.name)

        n_needed = max(0, TARGET_COUNT - original_count)
        generated = []
        if n_needed > 0:
            # Preload every source waveform once per species, not once per
            # augmented file - a species needing e.g. 120 new files from
            # only 30 sources would otherwise redecode the same handful of
            # files up to 4x each for no reason.
            waveforms = []
            for f in src_files:
                y, sr = sf.read(f, dtype="float32")
                # soundfile returns multichannel audio as (samples,
                # channels) - audiomentations requires mono as a plain 1D
                # array, so collapse to mono here. Consistent with the
                # rest of this pipeline (dataset.py's own SpectrogramDataset
                # loads audio as mono too) - species classification from a
                # stereo pair adds nothing here, both channels are the same
                # microphone recording.
                if y.ndim > 1:
                    y = y.mean(axis=1)
                waveforms.append((f, y, sr))

            chain = build_chain()
            for i in range(n_needed):
                # Round-robin through source files rather than random
                # sampling with replacement, so every original file
                # contributes roughly equally to the augmented set instead
                # of chance over-favouring some sources over others.
                src_file, y, sr = waveforms[i % len(waveforms)]
                y_aug = chain(samples=y, sample_rate=sr)
                out_name = f"{src_file.stem}__aug{i:03d}.wav"
                sf.write(dest_dir / out_name, y_aug, sr)
                generated.append({"output": out_name, "source": src_file.name})

        manifest["species"][species] = {
            "original_count": original_count,
            "generated_count": len(generated),
            "final_count": original_count + len(generated),
            "generated_files": generated,
        }
        print(f"{species}: {original_count} -> {original_count + len(generated)} (+{len(generated)} augmented)")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {MANIFEST_PATH}")
    print(f"Expanded training set written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
