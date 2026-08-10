# augmentation

Reproducible PyTorch/Hydra training pipeline for Project Echo species
classification (Sprint 1 baseline). Ported from an archived prototype
into its permanent home here, with two bug fixes, a rewritten
`augment.py`, and small config gaps closed - see Known Limitations below.

## Attribution

The Hydra config set (`config.yaml`, `config/model/*.yaml`,
`config/teacher_model/*.yaml`, `config/augmentation/*.yaml`,
`config/local/cpu.yaml`) was provided by **Kiernan Nguyen** for the
SpecAugment validation experiment work, and is used here per team
agreement to build on top of it rather than start a separate config
setup. This README and the ported/rewritten Python source are new for
this task.

## File Layout

```text
src/prototypes/engine/augmentation/
|-- main.py            Hydra entrypoint - dataset split, model build, training loop trigger
|-- train.py            Trainer class - AMP training loop, metric learning, distillation, TensorBoard
|-- dataset.py            SpectrogramDataset / index_directory - audio loading, caching, chunking
|-- augment.py             SpecAugment - spectrogram masking (REWRITTEN, see Known Limitations)
|-- model/                  Model architectures (EfficientNetV2, GhostEfficientNetV2, PANNs variants)
|-- config/                  Hydra config (see Attribution) - config.yaml, model/, teacher_model/, augmentation/, local/
|-- docs/training.md          Detailed training-loop walkthrough
|-- pyproject.toml             Scoped dependency manifest (uv-managed)
`-- uv.lock                     Locked dependency versions
```

## Setup and Execution Instructions

```powershell
cd src\prototypes\engine\augmentation
uv sync
```

This creates a local `.venv/` with everything the pipeline needs (see
Unresolved Dependencies below for what's *not* pinned yet).

Example runs (from this folder):

```powershell
# Default run (efficientnet_v2, default augmentation preset)
uv run python main.py

# Lighter/faster model, heavier augmentation preset
uv run python main.py model=ghost_efficientnet_v2 augmentation=heavy

# Windows/CPU-only dev machine (skips pretrained-weights download, fixes
# Hydra's working-directory behaviour so relative paths resolve) - see
# config/local/cpu.yaml for what this bundles
uv run python main.py +local=cpu

# Point at the real local dataset already checked out in this repo at
# src/prototypes/data_files/ (128 species, untracked - see .gitignore),
# combined with the CPU bundle above
uv run python main.py +local=cpu +local=local_data_files
```

Any `config.yaml` key can be overridden the same way, e.g.
`training.epochs=50`, `training.distillation.enabled=false`.

### Verify it works

```powershell
python src\tests\pipeline\engine_training\smoke_test\test_train_smoke.py
```

Runs the real pipeline end-to-end against a small synthetic dataset (no
real audio needed) - see `src/tests/pipeline/engine_training/smoke_test/README.md`.

## Baseline Configuration Record

- **Seed**: `training.seed=0` (`config.yaml`) now seeds `torch`, `numpy`,
  and stdlib `random` together (see Known Limitations - this was fixed
  as part of this task; only `torch` was seeded originally).
- **Split**: `train_split=0.8` / `val_split=0.2`, per-class stratified
  (`main.py`). No test split is currently wired up. **Each class needs
  at least 5 files** for the validation split to be non-empty - fewer
  files per class rounds `int(n * 0.2)` down to 0 and crashes
  `train.py`'s `_evaluate` with `ZeroDivisionError` (found while building
  the smoke test).
- **Preprocessing** (`config.yaml`'s `data:` block): sample_rate 48000,
  n_fft 4096, hop_length 480, n_mels 384, fmin 50, fmax 14000, top_db 80,
  clip duration 2s.
- **Core training params**: batch_size 64, grad_accum_steps 2, epochs
  500 (early-stopping patience 15), AdamW lr 1e-3, ReduceLROnPlateau,
  CircleLoss metric learning enabled by default (`use_arcface: circle`).
- **Baseline training log**: `docs/baseline_smoke_training_log.txt` -
  full stdout/stderr from a real (successful) run of the smoke test's
  exact scenario (3 synthetic classes, `ghost_efficientnet_v2`, 2
  epochs). A full run against the real ~900MB local dataset was not
  attempted this sprint - see Known Limitations.

## Unresolved Dependencies

- `scikit-learn` and `tensorboard` are used directly (`train.py`) but
  were only available transitively in the original archived
  `pyproject.toml`; both are now listed as direct dependencies here.
- `torch-audiomentations` and `lmdb` Windows-availability: installed and
  smoke-tested successfully on this Windows dev machine via `uv sync`,
  but worth re-confirming on the next contributor's machine given this is
  a Windows-heavy dev team (`config/local/cpu.yaml` exists for exactly
  this reason).
- The archive's original `requirements.txt` was deliberately **not**
  carried over - it pinned a conflicting TensorFlow/librosa version and
  was missing several hard dependencies (hydra-core, omegaconf, lmdb,
  audiomentations, torch-audiomentations, scikit-learn). This folder's
  `pyproject.toml`/`uv.lock` is the source of truth going forward.
- This is a deliberately separate Python environment from the repo-root
  `requirements.txt` files (which serve `src/production/*`, TF-based) -
  don't try to merge them.

## Known Limitations

- **This is a new baseline, not a reproduction of any previously-trained
  model.** The repo has three different "previously trained" models with
  conflicting/undocumented provenance; this pipeline was deliberately
  built on Kiernan's config as the ground-truth baseline going forward,
  per team decision, rather than reverse-engineering any of the old
  models' actual (largely unseeded, undocumented) training runs. Don't
  expect this pipeline's output to match any existing model's weights or
  accuracy.
- **`augment.py` was rewritten, not ported.** The archived version's
  `SpecAugment` constructor didn't match 4 of the 5 augmentation presets
  already committed to `config/augmentation/` (would raise `TypeError`
  on `hydra.utils.instantiate`). The rewritten version supports both the
  ratio-based API (`freq_mask_ratio`/`time_mask_ratio`, used by
  default/light/heavy) and the legacy pixel API
  (`freq_mask_param`/`time_mask_param`, used only by
  `original_unfixed_reference`) without changing any preset YAML. See
  `docs/training.md`'s "Spectrogram Augmentation" section.
- **`config/model/{ghost_efficientnet_v2,panns_cnn14,panns_mobilenetv1,panns_mobilenetv2}.yaml`
  were missing a required `norm_choice` key** (`model/__init__.py` reads
  `cfg.model.norm_choice` unconditionally) - all four would crash
  immediately on model construction. Added `norm_choice: freeze_bn` to
  each (matching what `efficientnet_v2.yaml`/`efficientnet_v2_qat.yaml`
  already used) as an additive fix, found while running the manual
  end-to-end check for this task. Flagging for Kiernan/Praveen to
  confirm `freeze_bn` is the intended choice per architecture, since
  `panns_mobilenetv2_qat.yaml` uses `swap_rms_norm` instead for its QAT
  variant - it's not automatically the same choice for every model.
- **The 6 `config/teacher_model/*.yaml` presets have the same missing
  `norm_choice` gap**, not fixed here since the distillation path is
  deferred (see below) and only exercises `teacher_model` when
  `training.distillation.enabled=true`.
- **Augmentation masking reproducibility**: `main.py` now seeds `torch`,
  `numpy`, and `random` together (previously only `torch`), since
  `augment.py`'s masking uses `random.randint`/`random.random`. This
  closes what would otherwise be a real gap in the "reproducible
  pipeline" framing of this task.
- **Two known bugs fixed during the port** (both in currently-unexercised
  code paths, zero risk to the default training path): `main.py` had a
  typo (`cfg.run.cKLDivLossheckpoint_path`) breaking the
  `run.test=true`/checkpoint-loading branch; `train.py`'s
  `save_checkpoint`/`load_checkpoint` referenced `OmegaConf` without
  importing it.
- **Explicitly deferred to Sprint 2** (not blockers for this task):
  - The `run.test=true` checkpoint-loading path (bug-fixed but unexercised).
  - The distillation path (`training.distillation.enabled=true`) - needs
    a teacher checkpoint that doesn't exist yet, plus the `norm_choice`
    gap in `config/teacher_model/*.yaml` noted above.
  - QAT preset variants (`model/quant.py` is ported and used by the
    `*_qat`/`ghost_efficientnet_v2` presets' `use_qat: true` path, which
    the smoke test does exercise via `ghost_efficientnet_v2`, but
    `trainer.model.quantise()` post-training conversion is not).
  - A full training run against the real local dataset
    (`src/prototypes/data_files/`, ~900MB, 128 species) to convergence -
    only a 2-epoch synthetic-data smoke run was done this sprint.
