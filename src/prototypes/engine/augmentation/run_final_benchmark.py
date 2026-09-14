"""
Runs training on the FINAL balanced dataset (Manisha/Raveesha's delivery -
Project-Echo/models-and-data/final_data_files/{train,validation,test}/),
once per augmentation preset - the "final" benchmark Required Work #3 asks
for, as distinct from run_augmentation_benchmark.py's current-dataset
fallback/original runs.

Why this isn't just main.py: main.py (reproducible_training_pipeline/)
expects ONE directory and does its own internal random 80/20 split
(main.py:67-96). That's wrong for pre-split data - pointing it at train/
alone would carve ANOTHER 20% out as a fake validation set (wasting
balanced training data) and never touch the real validation/ split
Manisha/Raveesha built at all. The whole point of their recording-level
split was to prevent leakage; re-splitting on top of it would undo that.

So this is a separate, parallel entrypoint - not a modification to
main.py/train.py (Nolan's pipeline, Sprint 2 §3.12) - that builds
train_dataset/val_dataset directly from the two pre-split folders (no
internal splitting at all) and hands them to the *same* Trainer/Model
classes main.py uses, unchanged. Same "reuse the real classes, don't
reimplement" principle as evaluate_checkpoint() in
run_augmentation_benchmark.py.

Needs to be a real @hydra.main entrypoint (like main.py), not the lighter
hydra.compose() API evaluate_checkpoint() uses - Trainer.__init__ calls
hydra.core.hydra_config.HydraConfig.get(), which only works inside an
actual Hydra-launched job, not a bare compose() call.
"""

import sys
from pathlib import Path

# dataset.py/model/train.py live in reproducible_training_pipeline/, a
# sibling folder to this one - add it to sys.path before importing from it.
PIPELINE_DIR = Path(__file__).resolve().parent.parent / "reproducible_training_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import SpectrogramDataset, index_directory, validation_collate_fn
from model import Model
from train import Trainer

# Relative to this file, since @hydra.main resolves config_path relative to
# the decorated function's own module location (same convention main.py
# uses with its own "config", just one level further up).
CONFIG_PATH = "../reproducible_training_pipeline/config"


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Same seeding as main.py:28-30 - torch, numpy, and stdlib random, so
    # augment.py's masking (random.randint/random.random) is reproducible
    # too, not just model init.
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    # cfg.system.train_dir / cfg.system.val_dir are new keys (not in the
    # checked-in config.yaml) - passed as CLI overrides, since there's no
    # existing config group that fits "point at two pre-split folders" the
    # way +local bundles fit "point at one folder".
    train_files, train_labels, train_class_names = index_directory(cfg.system.train_dir)
    val_files, val_labels, val_class_names = index_directory(cfg.system.val_dir)

    # Both splits must agree on the species set - if they don't, class
    # indices would silently mean different species in train vs val.
    if train_class_names != val_class_names:
        train_only = set(train_class_names) - set(val_class_names)
        val_only = set(val_class_names) - set(train_class_names)
        raise ValueError(
            f"train/validation species mismatch - train only: {train_only}, "
            f"validation only: {val_only}"
        )
    class_names = train_class_names

    print(f"Train: {len(train_files)} files, Validation: {len(val_files)} files, {len(class_names)} species")

    OmegaConf.set_struct(cfg, False)
    cfg.data.num_classes = len(class_names)
    OmegaConf.set_struct(cfg, True)

    audio_transforms = (
        hydra.utils.instantiate(cfg.augmentations.audio)
        if "augmentations" in cfg and "audio" in cfg.augmentations
        else None
    )
    image_transforms = (
        hydra.utils.instantiate(cfg.augmentations.image)
        if "augmentations" in cfg and "image" in cfg.augmentations
        else None
    )

    # Augmentation only on the training split - same rule as main.py,
    # confirmed in Required Work #2's review: validation never gets
    # image_transforms/audio_transforms.
    train_dataset = SpectrogramDataset(
        train_files, train_labels, cfg,
        audio_transforms=audio_transforms, image_transforms=image_transforms,
        is_train=True,
    )
    val_dataset = SpectrogramDataset(
        val_files, val_labels, cfg,
        audio_transforms=None, image_transforms=None,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=validation_collate_fn,
        pin_memory=True,
    )

    model = Model(cfg).to(device)
    print(model.summary())
    trainer = Trainer(cfg, model, train_loader, val_loader, device, cfg.model.name)
    trainer.train()


if __name__ == "__main__":
    main()
