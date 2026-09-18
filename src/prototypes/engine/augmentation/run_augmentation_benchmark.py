"""
Runs training once per augmentation preset (and once per seed - see
--seeds), so the results can be compared side by side. Supports two
dataset modes via --dataset:

  --dataset current  (default) - the small experiment_data_subset/ built
    from the current (unbalanced) dataset by build_experiment_subset.py.
    Launches reproducible_training_pipeline/main.py, which does its own
    internal random 80/20 train/val split from one folder. This is
    Required Work #1's "start now, don't wait for the balanced dataset"
    fallback/original run.

  --dataset balanced - the FINAL balanced dataset delivered by
    Manisha/Raveesha (Project-Echo/models-and-data/final_data_files/,
    pre-split into train/validation/test at the source-recording level to
    avoid leakage). Launches run_final_benchmark.py (this folder) instead
    of main.py, since main.py's internal-split logic would be wrong here -
    it would carve another 20% out of the balanced training set as a fake
    validation set and never touch the real validation/ split at all. This
    is Required Work #3's "final" benchmark.

Five presets, run in both modes:
  1. none - no augmentation at all
  2. original_unfixed_reference - the old SpecAugment settings (last trimester)
  3. light - gentle masking
  4. default - the newly fixed SpecAugment settings
  5. heavy - aggressive masking

Originally two separate scripts (run_augmentation_benchmark.py +
run_final_benchmark.py as a standalone orchestrator-less entrypoint);
merged into one so the orchestration/metrics-extraction logic - which has
already had two real bugs this sprint (stdout vs stderr, sys.path) - only
exists in one place instead of two that could silently drift apart.

Results are written to experiment_results/<arm_name>/ (--dataset current)
or experiment_results_balanced/<arm_name>/ (--dataset balanced), one file
per seed - kept in separate folders so a balanced-dataset run can never
overwrite a current-dataset one or vice versa.
"""

# argparse: lets --epochs and --batch-size be set from the command line
# instead of hardcoded, so this script can be reused for both a quick 1-epoch
# smoke test and the real multi-epoch experiment without editing the file.
import argparse

# re: used to pull the final epoch's metrics (train_loss, train_acc,
# val_loss, val_acc) out of main.py's printed training output.
import re

# shutil: used to move the checkpoint and TensorBoard log main.py produces
# into a per-arm results folder before the next arm's run overwrites them.
import shutil

# statistics: mean/spread across seeds for the summary table - population
# stdev (pstdev), since the seeds run are the entire set being reported,
# not a sample of some larger population.
import statistics

# subprocess: used to actually invoke `python main.py ...` (or
# run_final_benchmark.py) as a separate process for each arm, exactly as
# if it had been typed at the command line.
import subprocess

# sys: used to read sys.executable, so the subprocess calls reuse whichever
# Python interpreter is running this script (projectecho's), rather than
# risking a different one being picked up from PATH.
import sys

# Path: for building filesystem paths in a way that works correctly
# regardless of the OS's path separator conventions.
from pathlib import Path

# The folder this script lives in (src/prototypes/engine/augmentation/) -
# used for the data subset and for archiving results, but no longer where
# main.py runs from (see PIPELINE_DIR below).
SCRIPT_DIR = Path(__file__).resolve().parent

# main.py, train.py, dataset.py, and model/ no longer live in this folder -
# Nolan owns them in reproducible_training_pipeline/ (Sprint 2 §3.12), which
# also has a fixed checkpoint-path bug and a reproducibility fix (seeding
# numpy/random, not just torch) that this folder's old copies didn't have.
# Every subprocess call below runs with this as its working directory
# regardless of dataset mode (both main.py and run_final_benchmark.py
# resolve their own config path relative to their own file location, not
# cwd, so this is safe for either).
PIPELINE_DIR = SCRIPT_DIR.parent / "reproducible_training_pipeline"

# The final balanced dataset (Manisha/Raveesha, Required Work #3). Four
# `.parent`s reach the repo root (Project-Echo/) from SCRIPT_DIR - one
# fewer than build_experiment_subset.py's SOURCE_DIR, since that applies
# its `.parent`s to the raw Path(__file__) (which still includes the
# filename); SCRIPT_DIR has already had the filename stripped by its own
# first `.parent` above, so reusing "five" here would overshoot by one
# level - caught by actually checking FINAL_TRAIN_DIR.exists() below,
# not assumed.
FINAL_DATASET_DIR = SCRIPT_DIR.parent.parent.parent.parent / "models-and-data" / "final_data_files"
FINAL_TRAIN_DIR = FINAL_DATASET_DIR / "train"
FINAL_VAL_DIR = FINAL_DATASET_DIR / "validation"

# Where each arm's log, checkpoint, and TensorBoard file get archived -
# separate folders per dataset mode so one can never overwrite the other.
RESULTS_DIR = SCRIPT_DIR / "experiment_results"
RESULTS_DIR_BALANCED = SCRIPT_DIR / "experiment_results_balanced"

# The five experiment arms: (folder-safe name for results, augmentation
# preset to select via Hydra's `augmentation=<name>` override). Order matches
# the order they'll run in and appear in the summary table. Extended from
# Sprint 1's three arms (none/original_unfixed/default) to all five presets
# per sprint_2_augmentation_comparison_plan.md's "Concrete next steps".
ARMS = [
    ("none", "none"),
    ("original_unfixed", "original_unfixed_reference"),
    ("light", "light"),
    ("default", "default"),
    ("heavy", "heavy"),
]

# Regex to find main.py's final-epoch metrics line in its captured output.
# train.py prints (and tqdm reprints every step) a line containing all four
# of these values together, e.g.:
#   "...train_loss=24.3356, train_acc=0.0000, val_loss=7.2973, val_acc=0.1400}"
# Since tqdm reprints this line on every step, taking the LAST match found
# in the whole captured log gives the final epoch's metrics.
METRICS_PATTERN = re.compile(
    r"train_loss=([\d.]+).*?train_acc=([\d.]+).*?val_loss=([\d.]+).*?val_acc=([\d.]+)"
)


def build_command(python_exe, augmentation_preset, epochs, batch_size, seed, dataset):
    """Builds the command line for one experiment arm.

    Every override below exists for a specific, previously-diagnosed reason -
    see the inline comment on each one."""
    command = [
        python_exe,
        # --dataset current launches main.py, which does its own internal
        # random 80/20 split from one folder. --dataset balanced launches
        # run_final_benchmark.py instead - main.py's split logic would be
        # wrong on pre-split data (see this file's module docstring) - an
        # absolute path, since cwd is PIPELINE_DIR, not this script's folder.
        "main.py" if dataset == "current" else str(SCRIPT_DIR / "run_final_benchmark.py"),
        # Bundles two environment-specific workarounds into one opt-in
        # config file (config/local/cpu.yaml) instead of listing them
        # separately here: hydra.run.dir=. (keeps Hydra running in this
        # folder instead of switching into a new outputs/<date>/<time>/
        # working directory - without it, the relative
        # `experiment_data_subset` path below would not resolve) and
        # model.params.pretrained=false (skips a pretrained-weights
        # download that hits a Windows SSL certificate-store bug in this
        # setup). See that file for the full explanation of each.
        # (training.device=cpu used to be a third override here too, but
        # it's redundant: main.py already falls back to CPU automatically
        # via torch.cuda.is_available() whenever no CUDA build of PyTorch
        # is installed, which is the case here - so it was dropped
        # entirely instead of being moved into the bundle.)
        "+local=cpu",
    ]

    if dataset == "current":
        # Points at the small subset instead of the `b3` folder the
        # checked-in config defaults to, which does not exist here.
        # Absolute path (not a bare "experiment_data_subset") because
        # main.py runs with PIPELINE_DIR as its cwd, not this folder - a
        # relative path here would resolve against the wrong directory.
        command.append(f"system.audio_data_directory={SCRIPT_DIR / 'experiment_data_subset'}")
    else:
        # +system.train_dir / +system.val_dir are new keys (not in the
        # checked-in config.yaml) - run_final_benchmark.py reads these
        # directly instead of doing its own split.
        command.append(f"+system.train_dir={FINAL_TRAIN_DIR}")
        command.append(f"+system.val_dir={FINAL_VAL_DIR}")

    command += [
        # Removes the audio-level augmentation section entirely (the `~`
        # prefix is Hydra's "delete this key" syntax). It depends on a
        # `background_noise_dir` that doesn't exist in this environment, and
        # audio-level augmentation is outside this task's scope, which is
        # specifically about SpecAugment (the image/spectrogram-level
        # augmentation). Can't be folded into config/local/cpu.yaml - a
        # deletion override only works on the command line, not stored
        # inside a config file's content.
        "~augmentations.audio",
        # Caching OFF - kept off deliberately, not just inherited from
        # Sprint 1. Tried turning it on to avoid re-decoding audio every
        # epoch, but with num_workers=0, train_dataset and val_dataset are
        # two separate SpectrogramDataset objects sharing one process, and
        # each lazily opens its own LMDB handle on the same .cache path
        # (dataset.py:88-97) the first time it's used - the second one to
        # open (val, right after training's first epoch) crashes with
        # "lmdb.Error: The environment '.cache' is already open in this
        # process." This is a real, pre-existing bug in dataset.py, not
        # something introduced here - it just never triggered before
        # because caching was always off. Switching to CUDA (see
        # PIPELINE_DIR/training.device in config.yaml) made this moot
        # anyway: 137 training batches completed in ~75 seconds before this
        # bug was hit, roughly 15x faster than the CPU pace that motivated
        # trying caching in the first place.
        "system.use_disk_cache=False",
        # Ceiling on epochs, not a target - real run length comes from
        # training.early_stopping_patience=15 (config.yaml, applied
        # automatically in train.py) per
        # sprint_2_augmentation_comparison_plan.md, not this number. See
        # --epochs' own help text in main() for why.
        f"training.epochs={epochs}",
        # Passed in via --batch-size; larger than the checked-in default of
        # 8 to reduce the number of training steps per epoch on this
        # CPU-only setup.
        f"training.batch_size={batch_size}",
        # 0 = load data on the main process only, no worker subprocesses.
        # Simpler and more reliable than multiprocess workers for a one-off
        # small experiment on Windows.
        "training.num_workers=0",
        # Passed in via --seed, so a single arm can be re-run across
        # multiple seeds (per sprint_2_augmentation_comparison_plan.md -
        # "a single run of any size is still one data point"). Also seeds
        # numpy/random (main.py:29-30), so SpecAugment's masking is
        # reproducible per seed, not just model init.
        f"training.seed={seed}",
        # The one override that actually differs between arms of the same
        # seed - everything above is identical for all of them.
        f"augmentation={augmentation_preset}",
    ]
    return command


def evaluate_checkpoint(checkpoint_path, augmentation_preset, seed, dataset, split="validation"):
    """Independently computes macro-F1 and per-species F1 from a trained
    checkpoint.

    Needed because Trainer.test() (reproducible_training_pipeline/train.py)
    isn't wired into main.py's default run.train=true path - `if
    cfg.run.train` and `if cfg.run.test` are two separate `if` blocks in
    main.py, not if/elif, and cfg.run.test defaults to False - and even
    when it does run, it only returns aggregate weighted-average metrics
    with no per-species breakdown (train.py:349-361). Deliberately does not
    import from train.py/main.py to avoid depending on Nolan's Sprint 2
    changes there - only from the lower-level dataset.py/model/ modules
    both of them already depend on.

    For --dataset current, mirrors main.py's per-class stratified split
    exactly (same seed, same algorithm) so this evaluates against the
    *exact* held-out set the checkpoint was actually validated against,
    not a fresh random split that could accidentally include
    training-seen files (`split` is ignored in this mode - there is only
    ever one reconstructed held-out set). For --dataset balanced, there's
    no split to reconstruct - Manisha/Raveesha's delivery already comes
    pre-split into validation/ and test/, so this just loads whichever one
    `split` names directly. `split="validation"` is what every checkpoint
    was actually selected against during training (this is the "test acc"
    reported everywhere else in this benchmark); `split="test"` points at
    the untouched held-out set instead, for a genuine validation-to-test
    generalisation-gap check - explicit and part of this function's own
    interface, not something a caller has to reach into module globals to
    get (that was tried once as a quick monkeypatch of FINAL_VAL_DIR and
    correctly called out as fragile: it happened to be safe only because
    validation/ and test/ turned out to share an identical, identically-
    sorted 79-species set, which nothing enforced by construction).

    Either way, mirrors Trainer.test()'s file-level chunk aggregation
    (train.py:311-352): each val_loader item is one file's stacked chunks
    (batch_size=1 + validation_collate_fn), so chunk logits are averaged
    before taking argmax to get one prediction per file, matching how the
    real pipeline scores a file.
    """
    import random as _random
    import sys as _sys
    from collections import defaultdict as _defaultdict

    import numpy as _np
    import torch as _torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader

    # dataset.py/model/ live in PIPELINE_DIR, not this script's own folder -
    # without this, `from dataset import ...` raises ModuleNotFoundError.
    # Real bug caught the hard way: this only worked in an earlier isolated
    # test because that test manually added PIPELINE_DIR to sys.path itself
    # - run_arm()'s actual call to this function never did, so every real
    # run silently failed here and was swallowed by run_arm()'s own
    # try/except, reporting "F1 evaluation failed" instead of crashing.
    if str(PIPELINE_DIR) not in _sys.path:
        _sys.path.insert(0, str(PIPELINE_DIR))

    from dataset import SpectrogramDataset, index_directory, validation_collate_fn
    from model import Model

    overrides = ["+local=cpu"]
    if dataset == "current":
        overrides.append(f"system.audio_data_directory={SCRIPT_DIR / 'experiment_data_subset'}")
    else:
        # split picks which of Manisha/Raveesha's pre-split folders to
        # evaluate against - "validation" (default, matches every other
        # accuracy/macro-F1 in this benchmark) or "test" (the untouched
        # held-out set, for a genuine generalisation-gap check). Built
        # from FINAL_DATASET_DIR directly rather than always using
        # FINAL_VAL_DIR, so this is a real parameter, not something a
        # caller has to override a module global to change.
        if split not in ("validation", "test"):
            raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
        overrides.append(f"+system.train_dir={FINAL_TRAIN_DIR}")
        overrides.append(f"+system.val_dir={FINAL_DATASET_DIR / split}")
    overrides += [
        "~augmentations.audio",
        # Caching OFF - see build_command()'s own comment for why (a real
        # pre-existing lmdb "already open in this process" bug in
        # dataset.py when two SpectrogramDataset instances share a
        # process). Not hit here specifically (this only constructs one
        # dataset), but kept consistent with the training run rather than
        # leaving a real, if narrower, gap.
        "system.use_disk_cache=False",
        f"augmentation={augmentation_preset}",
        f"training.seed={seed}",
    ]

    with initialize_config_dir(config_dir=str(PIPELINE_DIR / "config"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    # Same seeding as main.py:28-30, so this reproduces the exact same
    # stratified split main.py built when this checkpoint was trained
    # (--dataset current only - balanced mode has no split to reproduce).
    _torch.manual_seed(cfg.training.seed)
    _np.random.seed(cfg.training.seed)
    _random.seed(cfg.training.seed)

    if dataset == "current":
        audio_files, labels, class_names = index_directory(cfg.system.audio_data_directory)

        OmegaConf.set_struct(cfg, False)
        cfg.data.num_classes = len(class_names)
        OmegaConf.set_struct(cfg, True)

        # Exact same per-class stratified split as main.py:67-96
        # (train_indices not needed here, only val_indices).
        class_indices = _defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)

        val_indices = []
        for label, indices in class_indices.items():
            indices = _torch.tensor(indices)
            shuffled_class_indices = indices[_torch.randperm(len(indices))]
            n_val = int(len(indices) * cfg.data.val_split)
            val_indices.extend(shuffled_class_indices[:n_val].tolist())

        val_files = [audio_files[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
    else:
        # No split to reconstruct - validation/ already is the held-out
        # set (built at the source-recording level to avoid leakage).
        val_files, val_labels, class_names = index_directory(cfg.system.val_dir)

        OmegaConf.set_struct(cfg, False)
        cfg.data.num_classes = len(class_names)
        OmegaConf.set_struct(cfg, True)

    val_dataset = SpectrogramDataset(
        val_files, val_labels, cfg, audio_transforms=None, image_transforms=None,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, collate_fn=validation_collate_fn
    )

    device = _torch.device("cpu")
    model = Model(cfg).to(device)
    model.load_state_dict(_torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_labels, all_preds = [], []
    with _torch.no_grad():
        for inputs, lbls in val_loader:
            inputs = inputs.to(device)
            true_label = lbls[0].item()
            outputs = model(inputs)  # (num_chunks, num_classes)
            aggregated = outputs.mean(dim=0)
            pred = _torch.argmax(aggregated).item()
            all_labels.append(true_label)
            all_preds.append(pred)

    # accuracy computed here too (not taken from the training log's
    # last-epoch val_acc) so it and macro_f1 are scored on identical
    # predictions from identical model weights - the best checkpoint, not
    # whichever epoch training happened to end on. With early stopping
    # active, the last epoch is by definition one of the non-improving
    # epochs that triggered the stop, so it can be a materially worse
    # model than the best checkpoint - comparing a last-epoch accuracy
    # against a best-checkpoint macro_f1 would silently compare two
    # different sets of weights.
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(len(class_names))),
    )
    per_species_f1 = dict(zip(class_names, per_class_f1.tolist()))

    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_species_f1": per_species_f1}


def run_arm(python_exe, arm_name, augmentation_preset, epochs, batch_size, seed, dataset):
    """Runs one experiment arm end-to-end: launches main.py (or
    run_final_benchmark.py, see build_command()) as a subprocess, saves its
    full output, archives its checkpoint/log files, and extracts the
    final-epoch metrics for the summary table"""
    results_dir = RESULTS_DIR if dataset == "current" else RESULTS_DIR_BALANCED
    print(f"\n=== Running arm: {arm_name} (augmentation={augmentation_preset}, seed={seed}, dataset={dataset}) ===")

    command = build_command(python_exe, augmentation_preset, epochs, batch_size, seed, dataset)
    print("Command:", " ".join(command))

    # capture_output=True buffers the subprocess's stdout/stderr so we can
    # both save it to disk and scan it for the metrics line below; text=True
    # decodes it as a string instead of raw bytes.
    #
    # encoding="utf-8" is required: without it, text=True falls back to the
    # system locale encoding (cp1252 on this Windows setup), and main.py's
    # output contains real Unicode characters (tqdm's progress-bar block
    # characters, a warning message with an emoji) that cp1252 cannot
    # decode. That crashed a background reader thread inside Python's own
    # subprocess module on the first real run of this script, which then
    # left result.stdout/result.stderr as None instead of raising a clean
    # error - errors="replace" is a second safety net so that even an
    # unexpected byte sequence in the future gets substituted with a
    # placeholder character instead of crashing the whole experiment run.
    result = subprocess.run(
        command, cwd=PIPELINE_DIR, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )

    # Each arm gets its own results folder so nothing from one arm overwrites
    # another's evidence.
    arm_dir = results_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Save the full captured output regardless of success or failure, so a
    # crash can still be diagnosed later from the archived log.
    #
    # `or ""` on each side guards against stdout/stderr being None - this
    # happened for real the first time this script ran (see the encoding
    # fix above): a decode crash inside subprocess's reader thread left
    # result.stdout as None, and this line crashed with
    # "TypeError: can only concatenate str (not NoneType) to str" instead of
    # saving whatever output *was* captured. With the encoding fix this
    # shouldn't happen again, but keeping the guard means a future problem
    # fails with a readable log file instead of losing all captured output.
    # Seed-specific filename: arm_dir is shared across every seed of the
    # same arm (multi-seed runs), so a fixed name would let the next seed's
    # run silently overwrite this one's log.
    log_path = arm_dir / f"train_log_seed{seed}.txt"
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")

    if result.returncode != 0:
        # Non-zero exit code means main.py itself raised an error - report
        # it and skip the metrics extraction / file archiving below, since
        # there's nothing valid to archive.
        print(f"  FAILED (exit code {result.returncode}) - see {log_path}")
        return arm_name, None

    # main.py (and run_final_benchmark.py) always writes to the same fixed
    # checkpoint filename regardless of which arm is active, so it must be
    # moved into this arm's folder before the next arm's run overwrites it.
    # Written into PIPELINE_DIR, since that's where the subprocess actually
    # ran (its cwd, via +local=cpu's hydra.run.dir=.), not SCRIPT_DIR.
    checkpoint_dest = arm_dir / f"best_efficientnet_v2_seed{seed}.pth"
    checkpoint = PIPELINE_DIR / "best_efficientnet_v2.pth"
    if checkpoint.exists():
        shutil.move(str(checkpoint), str(checkpoint_dest))

    # Same reasoning for the TensorBoard event file train.py writes.
    for event_file in PIPELINE_DIR.glob("events.out.tfevents.*"):
        shutil.move(str(event_file), str(arm_dir / event_file.name))

    # Search for every match of the metrics pattern and keep only the last
    # one - that's the final epoch's numbers. tqdm writes its postfix
    # (where these values live) to stderr by default, not stdout - search
    # both (same combined text already written to log_path above), not
    # just result.stdout, or every run silently finds nothing here despite
    # training having genuinely worked (real bug: hit this for real after
    # switching to CUDA - 3 arms "completed" with saved checkpoints but no
    # metrics recovered, because this only checked result.stdout, which
    # never had them).
    matches = METRICS_PATTERN.findall((result.stdout or "") + (result.stderr or ""))
    if not matches:
        # Should not normally happen if the run succeeded, but guarding
        # against it explicitly rather than crashing on an index error.
        print("  Completed, but could not find metrics in the output.")
        return arm_name, None

    train_loss, train_acc, val_loss, val_acc = matches[-1]
    metrics = {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        # Last-epoch val_acc from the training log - kept for the training
        # curve/diagnostics, but NOT the "official" accuracy figure (see
        # "accuracy" below) since with early stopping active the last
        # epoch is one of the non-improving epochs that triggered the
        # stop, not necessarily the best model.
        "last_epoch_val_acc": float(val_acc),
    }

    # accuracy + macro-F1 / per-species F1, both computed from the same
    # best-checkpoint predictions in evaluate_checkpoint() - see that
    # function's docstring for why accuracy isn't taken from the training
    # log's last-epoch val_acc above.
    if checkpoint_dest.exists():
        try:
            f1_metrics = evaluate_checkpoint(checkpoint_dest, augmentation_preset, seed, dataset)
            metrics["accuracy"] = f1_metrics["accuracy"]
            metrics["macro_f1"] = f1_metrics["macro_f1"]
            metrics["per_species_f1"] = f1_metrics["per_species_f1"]
        except Exception as e:
            # A failure here shouldn't discard the training metrics already
            # captured above - report it and move on with what succeeded.
            print(f"  F1 evaluation failed: {e}")

    print(f"  Done: {metrics}")
    return arm_name, metrics


def main():
    # Reads --epochs and --batch-size from the command line (with sensible
    # defaults), rather than hardcoding them, so the same script serves both
    # a quick smoke test (--epochs 1) and the real benchmark.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["current", "balanced"],
        default="current",
        help=(
            "'current' (default) - the small experiment_data_subset/ from "
            "the current dataset (Required Work #1's fallback/original "
            "run). 'balanced' - the final dataset from Manisha/Raveesha "
            "(Required Work #3's final run) - see this file's module "
            "docstring for why these need different training entrypoints."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help=(
            "Ceiling on epochs per arm, not a target - matches config.yaml's own "
            "default. Per sprint_2_augmentation_comparison_plan.md, real epoch "
            "count should come from training.early_stopping_patience=15 "
            "(train.py:299 - stops once val_loss hasn't improved for 15 "
            "epochs) rather than a small fixed count picked up front, since a "
            "short run makes augmentation look worse than it is by cutting "
            "training off before its early-convergence penalty passes. Only "
            "override this lower for a quick smoke test, e.g. --epochs 1."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help=(
            "Comma-separated seeds, e.g. --seeds 0,1,2. Each arm runs once per "
            "seed, so results report mean +/- spread instead of a single run - "
            "per sprint_2_augmentation_comparison_plan.md: 'a single run of any "
            "size is still one data point.'"
        ),
    )
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    results_dir = RESULTS_DIR if args.dataset == "current" else RESULTS_DIR_BALANCED

    # Reuse whichever Python interpreter is currently running this script,
    # so the subprocess calls use the same (projectecho) environment.
    python_exe = sys.executable

    results_dir.mkdir(exist_ok=True)

    # Run every arm at every seed, collecting each (arm, seed) run's metrics
    # (or None if it failed) into a nested dict: all_metrics[arm_name][seed].
    all_metrics = {arm_name: {} for arm_name, _ in ARMS}
    for seed in seeds:
        for arm_name, preset in ARMS:
            name, metrics = run_arm(
                python_exe, arm_name, preset, args.epochs, args.batch_size, seed, args.dataset
            )
            all_metrics[name][seed] = metrics

    # Per-arm summary: mean +/- spread across seeds (population stdev - the
    # seeds run aren't a sample of some larger population, they're the
    # entire set of runs being reported), not just the last seed's numbers.
    # Falls back to a single value with no +/- when only one seed was run.
    print(f"\n=== Summary (dataset={args.dataset}, mean over seeds) ===")
    print(
        "accuracy/macro_f1 are both scored on the best checkpoint's "
        "predictions (see evaluate_checkpoint()); train_loss/val_loss are "
        "training-curve diagnostics from the last epoch, not the best "
        "checkpoint - the two are not necessarily the same epoch once "
        "early stopping is involved."
    )
    # Column widths sized for the widest possible value, "-123.4567+/-12.3456"
    # (20 chars) - narrower columns let a multi-seed +/- value run into the
    # next column with no separator, since :<N only pads, never truncates.
    print(f"{'Arm':<20}{'accuracy':<20}{'macro_f1':<20}{'train_loss':<20}{'val_loss':<20}")
    for arm_name, _ in ARMS:
        per_seed = [m for m in all_metrics[arm_name].values() if m is not None]
        if not per_seed:
            print(f"{arm_name:<20}FAILED (all seeds)")
            continue

        def fmt(key):
            values = [m[key] for m in per_seed if key in m]
            if not values:
                return "n/a"
            mean = statistics.fmean(values)
            if len(values) > 1:
                return f"{mean:.4f}+/-{statistics.pstdev(values):.4f}"
            return f"{mean:.4f}"

        print(
            f"{arm_name:<20}{fmt('accuracy'):<20}{fmt('macro_f1'):<20}"
            f"{fmt('train_loss'):<20}{fmt('val_loss'):<20}"
        )

    # Per-species F1, averaged across seeds, per arm - the class-imbalance-
    # sensitive detail the aggregate table above can't show.
    print("\n=== Per-species F1 (mean over seeds) ===")
    for arm_name, _ in ARMS:
        per_seed = [m for m in all_metrics[arm_name].values() if m is not None and "per_species_f1" in m]
        if not per_seed:
            continue
        print(f"\n--- {arm_name} ---")
        species = per_seed[0]["per_species_f1"].keys()
        for sp in species:
            values = [m["per_species_f1"][sp] for m in per_seed]
            print(f"  {sp:<30}{statistics.fmean(values):.4f}")


# Only run main() when this file is executed directly (e.g.
# `python run_augmentation_benchmark.py`), not if it were ever imported from
# somewhere else.
if __name__ == "__main__":
    main()
