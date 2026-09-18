"""Regression test for the LMDB cache collision bug in the ported PyTorch
training pipeline (src/prototypes/engine/reproducible_training_pipeline/).

train_dataset and val_dataset are separate SpectrogramDataset instances that
each lazily open their own lmdb.Environment. With training.num_workers=0 both
live in the same process, and lmdb refuses to open the same environment path
twice there ("The environment '...' is already open in this process") - this
crashed the first real (non-synthetic) full training run as soon as
validation started, because system.use_disk_cache defaults to True in
config.yaml and both datasets pointed at the same cache_directory.

test_train_smoke.py does not catch this: it explicitly passes
system.use_disk_cache=false, so it never exercises the cache code path at
all. This test uses the default (cache enabled) so a regression here would
be caught.
"""

from __future__ import annotations

import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
PIPELINE_DIR = REPO_ROOT / "src" / "prototypes" / "engine" / "reproducible_training_pipeline"
MAIN_PY = PIPELINE_DIR / "main.py"

CLASSES = ["synth_class_a", "synth_class_b", "synth_class_c"]
SAMPLE_RATE = 48000
FILES_PER_CLASS = 6  # >=5 needed so the default val_split=0.2 keeps >=1 sample/class
TIMEOUT_SECONDS = 300


def _venv_python() -> str:
    """Prefer the pipeline's own uv-managed venv (from `uv sync`); fall back
    to whatever interpreter is running this test if it's missing."""
    candidates = [
        PIPELINE_DIR / ".venv" / "Scripts" / "python.exe",  # Windows
        PIPELINE_DIR / ".venv" / "bin" / "python",  # POSIX
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _write_sine_wav(path: Path, seconds: float, freq: float) -> None:
    """Write a tiny mono 16-bit PCM sine-wave .wav using only the stdlib."""
    n_samples = int(seconds * SAMPLE_RATE)
    with wave.open(str(path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        frames = bytearray()
        for i in range(n_samples):
            value = int(3000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
            frames += struct.pack("<h", value)
        wav_file.writeframes(bytes(frames))


class DiskCacheRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not MAIN_PY.is_file():
            self.skipTest(f"main.py not found at {MAIN_PY}; is the pipeline ported yet?")

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_path = Path(self._tmp.name)

        self.data_dir = tmp_path / "data"
        self.noise_dir = tmp_path / "background_noise"
        self.outputs_dir = tmp_path / "outputs"
        self.cache_dir = tmp_path / "cache"

        for i, class_name in enumerate(CLASSES):
            class_dir = self.data_dir / class_name
            class_dir.mkdir(parents=True)
            for j in range(FILES_PER_CLASS):
                seconds = 2.5 if j % 2 == 0 else 1.0
                freq = 220.0 + 55.0 * i
                _write_sine_wav(class_dir / f"clip_{j}.wav", seconds, freq)

        self.noise_dir.mkdir(parents=True)
        for i in range(2):
            _write_sine_wav(self.noise_dir / f"noise_{i}.wav", 2.0, freq=60.0 + i)

    def _run_main(self, overrides: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            _venv_python(),
            "main.py",
            f"system.audio_data_directory={self.data_dir}",
            f"system.background_noise_dir={self.noise_dir}",
            f"system.cache_directory={self.cache_dir}",
            "system.use_disk_cache=true",  # the actual bug trigger - opposite of test_train_smoke.py
            f"hydra.run.dir={self.outputs_dir}",
            "training.num_workers=0",  # single-process: this is where the two lmdb.open() calls collided
            "training.device=cpu",
            "training.seed=0",
            "training.distillation.enabled=false",
            *overrides,
        ]
        try:
            return subprocess.run(
                cmd,
                cwd=PIPELINE_DIR,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            self.fail(
                f"Regression test exceeded {TIMEOUT_SECONDS}s; pipeline may be hung.\n"
                f"--- partial stdout ---\n{stdout[-4000:]}\n"
                f"--- partial stderr ---\n{stderr[-4000:]}"
            )

    def test_train_then_validate_with_disk_cache_enabled(self) -> None:
        """Train + validate in one process with caching on - must not raise
        lmdb.Error("... is already open in this process")."""
        result = self._run_main(
            [
                "model=ghost_efficientnet_v2",
                "training.epochs=1",
                "training.batch_size=2",
            ]
        )

        combined_output = result.stdout + result.stderr
        self.assertNotIn(
            "already open in this process",
            combined_output,
            "LMDB cache collision regression: train_dataset and val_dataset opened the "
            "same lmdb environment path in one process. See dataset.py's cache_path "
            "(should be split by is_train into separate subdirectories).",
        )
        self.assertEqual(
            result.returncode,
            0,
            "main.py exited non-zero with disk cache enabled.\n"
            f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}",
        )

        # Confirms the actual fix mechanism (separate train/ and val/ cache
        # subdirectories), not just an incidental pass.
        self.assertTrue((self.cache_dir / "train").is_dir(), "expected a train/ cache subdirectory")
        self.assertTrue((self.cache_dir / "val").is_dir(), "expected a val/ cache subdirectory")

        checkpoints = list(self.outputs_dir.glob("best_*.pth"))
        self.assertTrue(checkpoints, f"No best_*.pth checkpoint found under {self.outputs_dir}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
