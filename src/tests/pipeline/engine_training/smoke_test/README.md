# Training Pipeline Smoke Test

## Purpose

Checks that the ported PyTorch/Hydra training pipeline at
`src/prototypes/engine/augmentation/` runs end-to-end: dataset indexing,
augmentation, model build, one training + validation pass, and checkpoint
/ TensorBoard output.

It exercises the real `main.py` CLI, unmodified, against a small
synthetic dataset generated on the fly (stdlib `wave`, no real audio
files or third-party packages needed to build the fixtures). It does
**not** validate model accuracy, only that the pipeline runs and produces
the expected artifacts - see `src/prototypes/engine/augmentation/README.md`
for a real (non-synthetic) baseline run.

## Location and Structure

```text
src/tests/pipeline/engine_training/smoke_test/
|-- test_train_smoke.py
`-- README.md
```

No `fixtures/` folder - all synthetic audio is generated in `setUp()`
inside a `tempfile.TemporaryDirectory()` that cleans itself up.

## Prerequisites

- Python 3.11 (matches `src/prototypes/engine/augmentation/pyproject.toml`)
- The pipeline's dependencies installed via `uv sync`, run once from
  `src/prototypes/engine/augmentation/` (creates a local `.venv/` there)
- No GPU, Docker, or MongoDB required

The test file itself needs no third-party packages to run (synthetic
audio is written with the stdlib `wave` module) - only the subprocess it
spawns (the real `main.py`) needs the pipeline's dependencies, resolved
via that local `.venv/`.

## Run the Test

From the repository root, after `uv sync` has been run at least once in
`src/prototypes/engine/augmentation/`:

```powershell
python src\tests\pipeline\engine_training\smoke_test\test_train_smoke.py
```

Expected summary:

```text
Ran 1 test in ~40-60s
OK
```

If `.venv/` doesn't exist yet at `src/prototypes/engine/augmentation/`,
the test falls back to whatever interpreter ran it - which will fail
with `ModuleNotFoundError` for `hydra`/`torch`/etc. Run `uv sync` first.

## What It Checks

Runs `main.py model=ghost_efficientnet_v2 training.epochs=2
training.batch_size=2` (plus the fixed synthetic dataset paths) as a
subprocess, then asserts:

- The process exits `0` (stdout/stderr tail included in the failure
  message otherwise).
- A `best_*.pth` checkpoint exists under the run's output directory.
- A TensorBoard `events.out.tfevents.*` file exists alongside it.
- `class_names.txt` exists and lists exactly the 3 synthetic classes.

`ghost_efficientnet_v2` is used instead of `config.yaml`'s default
`efficientnet_v2` purely for CPU wall-clock speed in this test - it has
no `pretrained` ImageNet-weights download to skip either, unlike the
`efficientnet_v2*` presets.

The synthetic dataset alternates 2.5s and 1.0s clips per class (config's
`audio_clip_duration` is 2s) to exercise both the random-crop and
pad-by-repeat branches in `dataset.py`, and includes synthetic
background-noise files so the default augmentation preset's
`AddBackgroundNoise` transform is actually exercised rather than left
untested. Each synthetic class has 6 files so the default
`val_split=0.2` per-class stratified split keeps at least one validation
sample per class - fewer than 5 files/class produces an empty validation
set and a `ZeroDivisionError` in `train.py`'s `_evaluate` (found while
building this test; see the Known Limitations section in the pipeline's
own README).

## Troubleshooting

- `main.py not found`: the test skips itself with a clear message if the
  pipeline hasn't been ported to `src/prototypes/engine/augmentation/` yet.
- `ModuleNotFoundError` inside the subprocess output: run `uv sync` from
  `src/prototypes/engine/augmentation/`.
- Timeout (default 300s): CPU-only training of even a tiny synthetic set
  can be slow on constrained hardware; the failure message includes
  whatever stdout/stderr was captured before the timeout.
