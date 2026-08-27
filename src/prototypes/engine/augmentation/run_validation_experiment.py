"""
Runs three short training runs on the small experiment_data_subset/ dataset, one per augmentation setting, so the results can be compared side by side:
  1. none - no augmentation at all
  2. original_unfixed_reference - the old SpecAugment settings (last trimester)
  3. default - the newly fixed SpecAugment settings

Each run is a invocation of the project's main.py training
entrypoint (not a reimplementation) - the only thing that changes
between the three runs is which `augmentation` preset is selected.

Results (final-epoch metrics, full training log, and the saved checkpoint)
are written to experiment_results/<arm_name>/.
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

# subprocess: used to actually invoke `python main.py ...` as a separate
# process for each arm, exactly as if it had been typed at the command line.
import subprocess

# sys: used to read sys.executable, so the subprocess calls reuse whichever
# Python interpreter is running this script (projectecho's), rather than
# risking a different one being picked up from PATH.
import sys

# Path: for building filesystem paths in a way that works correctly
# regardless of the OS's path separator conventions.
from pathlib import Path

# The folder this script lives in (src/prototypes/engine/augmentation/) -
# every subprocess call needs to run with this as its working directory,
# since main.py expects to find config/, dataset.py, etc. relative to here.
SCRIPT_DIR = Path(__file__).resolve().parent

# Where each arm's log, checkpoint, and TensorBoard file get archived.
RESULTS_DIR = SCRIPT_DIR / "experiment_results"

# The three experiment arms: (folder-safe name for results, augmentation
# preset to select via Hydra's `augmentation=<name>` override). Order matches
# the order they'll run in and appear in the summary table.
ARMS = [
    ("none", "none"),
    ("original_unfixed", "original_unfixed_reference"),
    ("default", "default"),
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


def build_command(python_exe, augmentation_preset, epochs, batch_size):
    """Builds the main.py command line for one experiment arm.

    Every override below exists for a specific, previously-diagnosed reason -
    see the inline comment on each one."""
    return [
        python_exe,
        "main.py",
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
        # Points at the small 10-species subset instead of the `b3` folder
        # the checked-in config defaults to, which does not exist here.
        "system.audio_data_directory=experiment_data_subset",
        # Removes the audio-level augmentation section entirely (the `~`
        # prefix is Hydra's "delete this key" syntax). It depends on a
        # `background_noise_dir` that doesn't exist in this environment, and
        # audio-level augmentation is outside this task's scope, which is
        # specifically about SpecAugment (the image/spectrogram-level
        # augmentation). Can't be folded into config/local/cpu.yaml - a
        # deletion override only works on the command line, not stored
        # inside a config file's content.
        "~augmentations.audio",
        # The on-disk caching layer (lmdb) isn't needed for a one-off small
        # run and adds complexity (file locking) for no benefit here.
        "system.use_disk_cache=False",
        # How many training epochs each arm runs for - deliberately small
        # per Required Work #6's own wording ("small baseline experiment"),
        # passed in from the command line via --epochs.
        f"training.epochs={epochs}",
        # Passed in via --batch-size; larger than the checked-in default of
        # 8 to reduce the number of training steps per epoch on this
        # CPU-only setup.
        f"training.batch_size={batch_size}",
        # 0 = load data on the main process only, no worker subprocesses.
        # Simpler and more reliable than multiprocess workers for a one-off
        # small experiment on Windows.
        "training.num_workers=0",
        # The one override that actually differs between the three arms -
        # everything above is identical for all of them.
        f"augmentation={augmentation_preset}",
    ]


def run_arm(python_exe, arm_name, augmentation_preset, epochs, batch_size):
    """Runs one experiment arm end-to-end: launches main.py as a subprocess,
    saves its full output, archives its checkpoint/log files, and extracts
    the final-epoch metrics for the summary table"""
    print(f"\n=== Running arm: {arm_name} (augmentation={augmentation_preset}) ===")

    command = build_command(python_exe, augmentation_preset, epochs, batch_size)
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
        command, cwd=SCRIPT_DIR, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )

    # Each arm gets its own results folder so nothing from one arm overwrites
    # another's evidence.
    arm_dir = RESULTS_DIR / arm_name
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
    log_path = arm_dir / "train_log.txt"
    log_path.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")

    if result.returncode != 0:
        # Non-zero exit code means main.py itself raised an error - report
        # it and skip the metrics extraction / file archiving below, since
        # there's nothing valid to archive.
        print(f"  FAILED (exit code {result.returncode}) - see {log_path}")
        return arm_name, None

    # main.py always writes to the same fixed checkpoint filename regardless
    # of which arm is active, so it must be moved into this arm's folder
    # before the next arm's run overwrites it.
    checkpoint = SCRIPT_DIR / "best_efficientnet_v2.pth"
    if checkpoint.exists():
        shutil.move(str(checkpoint), str(arm_dir / "best_efficientnet_v2.pth"))

    # Same reasoning for the TensorBoard event file train.py writes.
    for event_file in SCRIPT_DIR.glob("events.out.tfevents.*"):
        shutil.move(str(event_file), str(arm_dir / event_file.name))

    # Search the captured stdout for every match of the metrics pattern and
    # keep only the last one - that's the final epoch's numbers.
    matches = METRICS_PATTERN.findall(result.stdout)
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
        "val_acc": float(val_acc),
    }
    print(f"  Done: {metrics}")
    return arm_name, metrics


def main():
    # Reads --epochs and --batch-size from the command line (with sensible
    # defaults), rather than hardcoding them, so the same script serves both
    # a quick smoke test (--epochs 1) and the real experiment (--epochs 3).
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epochs per arm - kept small deliberately, this is a validity check, not a benchmark",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Reuse whichever Python interpreter is currently running this script,
    # so the subprocess calls use the same (projectecho) environment.
    python_exe = sys.executable

    RESULTS_DIR.mkdir(exist_ok=True)

    # Run every arm in turn, collecting each one's final metrics (or None if
    # it failed) into a dict keyed by arm name for the summary table below.
    all_metrics = {}
    for arm_name, preset in ARMS:
        name, metrics = run_arm(python_exe, arm_name, preset, args.epochs, args.batch_size)
        all_metrics[name] = metrics

    # Print a simple aligned table comparing all three arms' final metrics -
    # this is what gets copied into the Required Work #6 results write-up.
    print("\n=== Summary (final epoch) ===")
    print(f"{'Arm':<20}{'train_loss':<12}{'train_acc':<12}{'val_loss':<12}{'val_acc':<10}")
    for arm_name, _ in ARMS:
        m = all_metrics.get(arm_name)
        if m is None:
            print(f"{arm_name:<20}FAILED")
        else:
            print(
                f"{arm_name:<20}{m['train_loss']:<12.4f}{m['train_acc']:<12.4f}"
                f"{m['val_loss']:<12.4f}{m['val_acc']:<10.4f}"
            )


# Only run main() when this file is executed directly (e.g.
# `python run_validation_experiment.py`), not if it were ever imported from
# somewhere else.
if __name__ == "__main__":
    main()