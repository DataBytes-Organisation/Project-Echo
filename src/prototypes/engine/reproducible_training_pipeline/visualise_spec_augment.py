"""
Generates before-and-after spectrogram examples.
Imports the real `SpecAugment` class from augment.py directly so it is not
reimplemented here.

Mel spectrogram settings below mirror config/config.yaml's `data:` section
(sample_rate, n_fft, hop_length, n_mels, fmin, fmax, top_db) so the pictures
represent what SpecAugment actually receives during training. Spectrograms
are computed with librosa (already in the projectecho environment) rather
than torchaudio (which the real training pipeline uses), purely to avoid
installing torchaudio for this visualisation step - values will be close but
not 100% the same. That does not affect what this script demonstrates, i.e., the
shape/extent of masking.
"""

import random
from pathlib import Path

import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch

from augment import SpecAugment

# --- Settings mirrored from config/config.yaml `data:` section ---
SAMPLE_RATE = 48000
N_FFT = 4096
HOP_LENGTH = 480
N_MELS = 384
FMIN = 50
FMAX = 14000
TOP_DB = 80
CLIP_DURATION = 2 # seconds - gives ~201 time frames, matching the review's finding
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# Four `.parent`s reach the repo root (Project-Echo/) from this file's folder
# (src/prototypes/engine/reproducible_training_pipeline/), then down into models-and-data/data_files
# - the real dataset's location in this repo.
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "models-and-data" / "data_files"
OUTPUT_DIR = Path(__file__).resolve().parent / "spectrogram_examples"

random.seed(42)


def load_clip(file_path):
    """Load audio, resample to SAMPLE_RATE, mix to mono, and force to exactly
    CLIP_SAMPLES long - repeat if shorter, random crop if longer - mirroring
    the approach dataset.py uses for training clips (see __getitem__)."""
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(y) < CLIP_SAMPLES:
        repeats = int(np.ceil(CLIP_SAMPLES / len(y)))
        y = np.tile(y, repeats)
    if len(y) > CLIP_SAMPLES:
        start = random.randint(0, len(y) - CLIP_SAMPLES)
        y = y[start : start + CLIP_SAMPLES]
    return y


def to_spectrogram(y):
    """Waveform -> normalised [0,1] spectrogram, matching dataset.py's steps:
    mel spectrogram -> dB -> normalise using top_db (dataset.py line 124).

    ref=np.max (not a fixed ref) is used so this clip's own loudest point
    maps to 0 dB - this guarantees a clean, valid [0,1] range after
    normalisation regardless of the clip's absolute loudness, which is what
    dataset.py's `(spec + top_db) / top_db` formula assumes as its input."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)
    mel_norm = (mel_db + TOP_DB) / TOP_DB  # [-top_db, 0] -> [0, 1]
    mel_norm = np.clip(mel_norm, 0.0, 1.0)
    return mel_norm  # shape (F, T) e.g. (384, 201)


def find_sample_files(n=4):
    """Pick n audio files from different species folders in data_files/."""
    species_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    random.shuffle(species_dirs)
    chosen = []
    for d in species_dirs:
        files = [f for f in d.iterdir() if f.suffix.lower() in (".wav", ".mp3", ".ogg", ".flac")]
        if files:
            chosen.append((d.name, random.choice(files)))
        if len(chosen) >= n:
            break
    return chosen


def render_panel(ax, spec, title):
    ax.imshow(spec, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    samples = find_sample_files(n=3)
    if not samples:
        print(f"No audio files found under {DATA_DIR}")
        return

    # p=1.0 on every augmenter below so masking always happens in these demo
    # images - the preset's own `p` only controls how OFTEN masking happens
    # during real training, not whether it CAN happen, so forcing it here
    # just makes sure every saved image actually shows a masked result.

    default_aug = SpecAugment(
        p=1.0,
        n_freq_mask=2,
        n_time_mask=2,
        freq_mask_ratio=0.15,
        time_mask_ratio=0.10,
        max_total_time_ratio=0.4,
        mask_value="zero",
    )

    # Reproduces the ORIGINAL, unfixed settings from the review (freq/time
    # mask widths as fixed pixel counts: 30 and 80) with the new safety cap
    # deliberately maxed out (max_total_time_ratio=1.0, i.e. no effective
    # cap) so this panel shows the ~80%-of-clip time-masking bug exactly as
    # it was found - this is NOT a recommended preset, it exists only to
    # visually document the bug for the review.
    original_buggy_aug = SpecAugment(
        p=1.0,
        n_freq_mask=2,
        n_time_mask=2,
        freq_mask_param=30,
        time_mask_param=80,
        max_total_time_ratio=1.0,
        mask_value="zero",
    )

    for species, file_path in samples:
        print(f"Processing {species}: {file_path.name}")
        y = load_clip(file_path)
        spec = to_spectrogram(y)  # (F, T) numpy, values in [0, 1]

        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)  # (1, F, T)

        default_masked = default_aug(spec_tensor.clone()).squeeze(0).numpy()
        buggy_masked = original_buggy_aug(spec_tensor.clone()).squeeze(0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        render_panel(axes[0], spec, f"{species}\nOriginal (no augmentation)")
        render_panel(axes[1], default_masked, "After SpecAugment\nwith new default preset")
        render_panel(axes[2], buggy_masked, "After SpecAugment\nwith existing hardcoded settings (last trimester)")
        fig.tight_layout()

        safe_name = species.replace(" ", "_")
        out_path = OUTPUT_DIR / f"{safe_name}_comparison.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  saved {out_path}")

    print(f"\nDone. {len(samples)} comparison images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
