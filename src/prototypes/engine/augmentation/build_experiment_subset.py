"""
Builds the small dataset subset used for running a small
baseline experiment to confirm that augmented training data remains valid.

Selects the N species with the most audio files in data_files/ (so every selected
species has enough clips for a meaningful train/val split), then copies the
first FILES_PER_SPECIES files (sorted by filename for determinism) from
each into a new subset folder.

Does not modify data_files/ - only copies files out of it.
Safe to re-run: existing copies are simply overwritten with identical
content.
"""

import shutil
from pathlib import Path

# ---Configuration---
N_SPECIES = 10 # number of species to include in the subset
FILES_PER_SPECIES = 100 # number of clips to copy per species
AUDIO_EXTENSIONS = (".wav", ".mp3", ".ogg", ".flac")

# Four `.parent`s reach the repo root (Project-Echo/) from this file's folder
# (src/prototypes/engine/augmentation/), then down into models-and-data/data_files
# - the real dataset's location in this repo.
SOURCE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "models-and-data" / "data_files"
DEST_DIR = Path(__file__).resolve().parent / "experiment_data_subset"


def count_audio_files(species_dir):
    """count audio files directly inside a species folder"""
    return sum(1 for f in species_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)


def select_top_species(n):
    """Pick the n species with the most audio files in SOURCE_DIR.
    Selecting by file count (rather than a hardcoded species list) means
    re-running this script after data_files/ changes will always pick
    species with enough clips for a sensible train/val split, and makes the
    selection reproducible from the data itself rather than a fixed list
    that could go stale."""
    species_dirs = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]
    counts = [(d, count_audio_files(d)) for d in species_dirs]
    counts.sort(key=lambda pair: pair[1], reverse=True)
    return counts[:n]


def copy_species_subset(species_dir, n_files):
    """copy the first n_files audio files (sorted by filename for a
    deterministic and reproducible selection) from species_dir into a matching
    folder under DEST_DIR"""
    dest = DEST_DIR / species_dir.name
    dest.mkdir(parents=True, exist_ok=True)

    files = sorted(
        f for f in species_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS
    )[:n_files]

    for f in files:
        shutil.copy2(f, dest / f.name)

    return len(files)


def main():
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_top_species(N_SPECIES)
    print(f"Selected {len(selected)} species with the most audio files:")
    for species_dir, count in selected:
        print(f"  {species_dir.name}: {count} files available")

    print(f"\nCopying up to {FILES_PER_SPECIES} files per species into {DEST_DIR} ...")
    total_copied = 0
    for species_dir, _ in selected:
        copied = copy_species_subset(species_dir, FILES_PER_SPECIES)
        total_copied += copied
        print(f"  {species_dir.name}: copied {copied} files")

    print(f"\nDone. {total_copied} files across {len(selected)} species in {DEST_DIR}")


if __name__ == "__main__":
    main()
