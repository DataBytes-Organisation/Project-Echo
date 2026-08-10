"""End-to-end smoke test for the ported PyTorch training pipeline
(src/prototypes/engine/augmentation/).

Runs the real `main.py` Hydra CLI, unmodified, against a small synthetic
dataset generated on the fly. Confirms the pipeline runs end-to-end
(dataset indexing, augmentation, model build, one training + validation
pass, checkpoint + TensorBoard output) without depending on the real
~900MB local species dataset. Does NOT validate model accuracy - only
that the pipeline runs and produces the expected artifacts.

No third-party Python packages are required to run this test file itself
(synthetic audio is written with the stdlib `wave` module); the pipeline
dependencies are only needed inside the subprocess this test spawns, via
the pipeline's own uv-managed virtual environment.
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
AUGMENTATION_DIR = REPO_ROOT / "src" / "prototypes" / "engine" / "augmentation"
MAIN_PY = AUGMENTATION_DIR / "main.py"

CLASSES = ["synth_class_a", "synth_class_b", "synth_class_c"]
SAMPLE_RATE = 48000
FILES_PER_CLASS = 6  # >=5 needed so the default val_split=0.2 keeps >=1 sample/class
LONG_CLIP_SECONDS = 2.5  # exercises the random-crop branch (clip_duration=2s in config.yaml)
SHORT_CLIP_SECONDS = 1.0  # exercises the pad-by-repeat branch
TIMEOUT_SECONDS = 300


def _venv_python() -> str:
    """Prefer the pipeline's own uv-managed venv (from `uv sync`); fall back
    to whatever interpreter is running this test if it's missing."""
    candidates = [
        AUGMENTATION_DIR / ".venv" / "Scripts" / "python.exe",  # Windows
        AUGMENTATION_DIR / ".venv" / "bin" / "python",  # POSIX
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


class TrainingPipelineSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        if not MAIN_PY.is_file():
            self.skipTest(f"main.py not found at {MAIN_PY}; is the pipeline ported yet?")

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp_path = Path(self._tmp.name)

        self.data_dir = tmp_path / "data"
        self.noise_dir = tmp_path / "background_noise"
        self.outputs_dir = tmp_path / "outputs"

        for i, class_name in enumerate(CLASSES):
            class_dir = self.data_dir / class_name
            class_dir.mkdir(parents=True)
            for j in range(FILES_PER_CLASS):
                # Alternate long/short clips to exercise both the
                # random-crop and pad-by-repeat branches in dataset.py.
                seconds = LONG_CLIP_SECONDS if j % 2 == 0 else SHORT_CLIP_SECONDS
                freq = 220.0 + 55.0 * i  # distinct tone per class, cosmetic only
                _write_sine_wav(class_dir / f"clip_{j}.wav", seconds, freq)

        self.noise_dir.mkdir(parents=True)
        for i in range(2):
            # Required because the default augmentation preset's
            # AddBackgroundNoise transform needs real files at
            # system.background_noise_dir, or that code path goes
            # untested by this smoke test.
            _write_sine_wav(self.noise_dir / f"noise_{i}.wav", 2.0, freq=60.0 + i)

    def _run_main(self, overrides: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            _venv_python(),
            "main.py",
            f"system.audio_data_directory={self.data_dir}",
            f"system.background_noise_dir={self.noise_dir}",
            "system.use_disk_cache=false",
            f"hydra.run.dir={self.outputs_dir}",
            "training.num_workers=0",
            "training.device=cpu",
            "training.seed=0",
            "training.distillation.enabled=false",
            *overrides,
        ]
        try:
            return subprocess.run(
                cmd,
                cwd=AUGMENTATION_DIR,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            self.fail(
                f"Smoke test exceeded {TIMEOUT_SECONDS}s; pipeline may be hung or "
                f"too slow for a CPU 2-epoch synthetic run.\n"
                f"--- partial stdout ---\n{stdout[-4000:]}\n"
                f"--- partial stderr ---\n{stderr[-4000:]}"
            )

    def test_training_pipeline_runs_end_to_end(self) -> None:
        result = self._run_main(
            [
                "model=ghost_efficientnet_v2",
                "training.epochs=2",
                "training.batch_size=2",
            ]
        )

        self.assertEqual(
            result.returncode,
            0,
            "main.py exited non-zero.\n"
            f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}",
        )

        checkpoints = list(self.outputs_dir.glob("best_*.pth"))
        self.assertTrue(checkpoints, f"No best_*.pth checkpoint found under {self.outputs_dir}")

        events = list(self.outputs_dir.glob("events.out.tfevents.*"))
        self.assertTrue(events, f"No TensorBoard event file found under {self.outputs_dir}")

        class_names_file = self.outputs_dir / "class_names.txt"
        self.assertTrue(class_names_file.is_file(), f"class_names.txt not found under {self.outputs_dir}")
        found_classes = set(class_names_file.read_text().split())
        self.assertEqual(found_classes, set(CLASSES))


if __name__ == "__main__":
    unittest.main(verbosity=2)
