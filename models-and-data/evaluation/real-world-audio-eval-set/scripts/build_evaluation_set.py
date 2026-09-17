#!/usr/bin/env python3
"""
Project Echo - Engine Team - Sriram Bharadwaz Miryalkar
Sprint 1 Task 3.14: Real-World Audio Evaluation-Set Preparation
Sprint 2 Task 3.14: Real-World Evaluation Set Finalisation

This script builds an organised, labelled real-world audio evaluation subset
from wildlife audio recordings already present in the Project Echo repository
(models-and-data/samples/store_audio/), derives controlled field-condition
variants (background noise, weak signal, overlapping calls) from those real
recordings, and verifies that every resulting file passes through the exact
audio-preprocessing function used by the production Engine
(EchoEngine.efficientnetv2_preprocess_audio_bytes in
src/production/engine/echo_engine.py), which is the currently ACTIVE_INFERENCE_MODEL
per src/production/engine/echo_engine.json.

Usage:
    python3 build_evaluation_set.py

Outputs (written under ../evaluation_set/):
    clean/                  - unmodified source recordings, resampled to WAV
    background_noise/       - source + calibrated pink noise at ~10 dB SNR
    weak_signal/             - source attenuated + calibrated noise at ~0 dB SNR
    overlapping_calls/       - two source species mixed together at ~0 dB relative level
    evaluation_metadata.csv  - one row per evaluation file
    evaluation_metadata.json - same content, machine-readable
    preprocessing_compatibility_report.md - real run results against the
                                             production preprocessing function
"""

import io
import json
import csv
import hashlib
import time
import traceback
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DELIVERABLE_DIR = SCRIPT_DIR.parent
RAW_AUDIO_DIR = Path("/home/claude/project-echo-work/raw_audio")
OUT_DIR = DELIVERABLE_DIR / "evaluation_set"
CLASS_MAPPING_PATH = SCRIPT_DIR / "class_mapping.json"          # copied from repo
PREPROCESS_CONFIG_PATH = SCRIPT_DIR / "preprocess_config.json"   # copied from repo

# ---------------------------------------------------------------------------
# Source recordings actually found in the Project Echo repository under
# models-and-data/samples/store_audio/ (git commit 01ebbd1, main branch).
# These are real, distinct wildlife/livestock species recordings (verified by
# SHA-256, they are NOT duplicates of one another), 22.05 kHz mono, 10s each.
# ---------------------------------------------------------------------------
SOURCE_SPECIES = {
    "Alauda_arvensis": {
        "file": "Alauda_arvensis.wav",
        "scientific_name": "Alauda arvensis",
        "common_name": "Eurasian Skylark",
        "repo_path": "models-and-data/samples/store_audio/Alauda Arvensis.wav",
    },
    "Capra_hircus": {
        "file": "Capra_hircus.wav",
        "scientific_name": "Capra Hircus",
        "common_name": "Domestic Goat",
        "repo_path": "models-and-data/samples/store_audio/Capra Hircus.wav",
    },
    "Cervus_unicolour": {
        "file": "Cervus_unicolour.wav",
        "scientific_name": "Cervus Unicolour",
        "common_name": "Sambar Deer",
        "repo_path": "models-and-data/samples/store_audio/Cervus Unicolour.wav",
    },
    "Pachycephala_rufiventris": {
        "file": "Pachycephala_rufiventris.wav",
        "scientific_name": "Pachycephala Rufiventris",
        "common_name": "Rufous Whistler",
        "repo_path": "models-and-data/samples/store_audio/Pachycephala Rufiventris.wav",
    },
    "Strepera_graculina": {
        "file": "Strepera_graculina.wav",
        "scientific_name": "Strepera Graculina",
        "common_name": "Pied Currawong",
        "repo_path": "models-and-data/samples/store_audio/Strepera Graculina.wav",
    },
    "Sus_scrofa": {
        "file": "Sus_scrofa.wav",
        "scientific_name": "Sus Scrofa",
        "common_name": "Wild Boar / Feral Pig",
        "repo_path": "models-and-data/samples/store_audio/Sus Scrofa.wav",
    },
}

OVERLAP_PAIRS = [
    ("Alauda_arvensis", "Capra_hircus"),
    ("Cervus_unicolour", "Pachycephala_rufiventris"),
    ("Strepera_graculina", "Sus_scrofa"),
]

# Condition-derivation parameters (documented, reproducible, deterministic).
RANDOM_SEED = 42
BACKGROUND_NOISE_SNR_DB = 10.0
WEAK_SIGNAL_SNR_DB = 0.0
WEAK_SIGNAL_ATTENUATION_DB = -12.0
OVERLAP_RELATIVE_LEVEL_DB = 0.0

WORKING_SR = 22050  # native sample rate of the source recordings


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def db_to_amplitude(db):
    return 10.0 ** (db / 20.0)


def pink_noise(n_samples, rng):
    """Generate pink (1/f) noise via spectral shaping of white noise."""
    white = rng.standard_normal(n_samples)
    fft = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    fft = fft / np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n_samples)
    pink = pink / (np.max(np.abs(pink)) + 1e-12)
    return pink.astype(np.float32)


def mix_at_snr(signal, noise, snr_db):
    """Scale `noise` so that the mix has the requested SNR relative to `signal`."""
    sig_rms = rms(signal)
    noise_rms = rms(noise) + 1e-12
    target_noise_rms = sig_rms / db_to_amplitude(snr_db)
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = signal + scaled_noise
    peak = np.max(np.abs(mixed))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def load_source(name):
    info = SOURCE_SPECIES[name]
    path = RAW_AUDIO_DIR / info["file"]
    audio, sr = librosa.load(str(path), sr=WORKING_SR, mono=True)
    return audio.astype(np.float32), sr


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Production Engine preprocessing function
# Ported verbatim (logic-for-logic) from:
#   src/production/engine/echo_engine.py
#   EchoEngine.efficientnetv2_preprocess_audio_bytes()
# This is the preprocessing path for ACTIVE_INFERENCE_MODEL =
# "efficientnetv2_tflite" as configured in src/production/engine/echo_engine.json.
# Reproduced here (rather than imported) because the live Engine class also
# requires a MongoDB connection, GCP credentials and a loaded TFLite
# interpreter at construction time, none of which are available in this
# evaluation-set-preparation context. The preprocessing math and control flow
# below are unchanged from the source file.
# ---------------------------------------------------------------------------
def load_preprocess_config():
    with open(PREPROCESS_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_class_mapping():
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def efficientnetv2_preprocess_audio_bytes(audio_bytes, preprocess_config):
    """Faithful port of the production Engine's preprocessing function."""
    audio_file = io.BytesIO(audio_bytes)

    target_sr = int(preprocess_config["target_sr"])
    duration_s = float(preprocess_config["duration_s"])
    n_mels = int(preprocess_config["n_mels"])
    hop_length = int(preprocess_config["hop_length"])
    fmin = float(preprocess_config["fmin"])
    fmax = float(preprocess_config["fmax"])

    audio, sample_rate = librosa.load(audio_file, sr=target_sr, mono=True)
    audio = audio.astype(np.float32)

    target_length = int(target_sr * duration_s)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_mels=n_mels, hop_length=hop_length,
        fmin=fmin, fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max).astype(np.float32)

    mel_mean = float(np.mean(mel_db))
    mel_std = float(np.std(mel_db))
    mel_db = (mel_db - mel_mean) / (mel_std + 1e-6)

    input_tensor = np.expand_dims(mel_db, axis=0)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # NCHW: [1,1,n_mels,time]

    return input_tensor, audio, sample_rate


# The TFLite model's actual input tensor shape, taken from
# src/production/engine/echo_engine.py docstring: [1, 1, 128, 313]
EXPECTED_TFLITE_INPUT_SHAPE = (1, 1, 128, 313)


def check_preprocessing_compatibility(wav_path, preprocess_config):
    """Run the file through the real preprocessing function and record results."""
    result = {
        "status": None,
        "error": None,
        "input_tensor_shape": None,
        "input_tensor_dtype": None,
        "matches_expected_tflite_shape": None,
        "contains_nan_or_inf": None,
        "value_min": None,
        "value_max": None,
        "processing_time_ms": None,
    }
    try:
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        t0 = time.perf_counter()
        tensor, audio, sr = efficientnetv2_preprocess_audio_bytes(
            audio_bytes, preprocess_config
        )
        t1 = time.perf_counter()

        result["status"] = "PASS"
        result["input_tensor_shape"] = list(tensor.shape)
        result["input_tensor_dtype"] = str(tensor.dtype)
        result["matches_expected_tflite_shape"] = (
            tuple(tensor.shape) == EXPECTED_TFLITE_INPUT_SHAPE
        )
        result["contains_nan_or_inf"] = bool(
            np.isnan(tensor).any() or np.isinf(tensor).any()
        )
        result["value_min"] = float(np.min(tensor))
        result["value_max"] = float(np.max(tensor))
        result["processing_time_ms"] = round((t1 - t0) * 1000.0, 2)
    except Exception as e:  # noqa: BLE001 - we want to record any failure mode
        result["status"] = "FAIL"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc(limit=3)
    return result


# ---------------------------------------------------------------------------
# Build the evaluation set
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    for sub in ["clean", "background_noise", "weak_signal", "overlapping_calls"]:
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    class_mapping = load_class_mapping()
    label_to_index = class_mapping["label_to_index"]
    preprocess_config = load_preprocess_config()

    # Also load the legacy combined_pipeline config values for reference (not
    # used for the pass/fail check, since ACTIVE_INFERENCE_MODEL is
    # efficientnetv2_tflite, but recorded for the compatibility report).
    with open(SCRIPT_DIR / "echo_engine_config.json", "r", encoding="utf-8") as f:
        legacy_config = json.load(f)

    metadata_rows = []
    compat_rows = []

    def label_lookup(scientific_name):
        idx = label_to_index.get(scientific_name)
        return idx

    def write_and_register(rel_path, audio, sr, row):
        out_path = OUT_DIR / rel_path
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
        row["duration_s"] = round(len(audio) / sr, 3)
        row["sample_rate_hz"] = sr
        row["channels"] = 1
        row["file_sha256"] = sha256_of_file(out_path)
        row["relative_path"] = rel_path
        compat = check_preprocessing_compatibility(out_path, preprocess_config)
        row.update({f"engine_preproc_{k}": v for k, v in compat.items() if k != "traceback"})
        metadata_rows.append(row)
        compat_rows.append((rel_path, compat))
        status = compat["status"]
        print(f"[{status}] {rel_path}")
        return out_path

    # ---- 1. Clean condition: unmodified source recordings ----
    clean_audio_cache = {}
    for name, info in SOURCE_SPECIES.items():
        audio, sr = load_source(name)
        clean_audio_cache[name] = (audio, sr)
        idx = label_lookup(info["scientific_name"])
        row = {
            "file_id": f"{name}__clean",
            "species_scientific_name": info["scientific_name"],
            "species_common_name": info["common_name"],
            "in_model_class_list": idx is not None,
            "model_class_index": idx,
            "condition_category": "clean",
            "derivation": "unmodified source recording (resampled to 22.05kHz mono PCM16 WAV)",
            "source_repo_path": info["repo_path"],
            "source_repo": "https://github.com/DataBytes-Organisation/Project-Echo",
            "source_note": (
                "Pre-existing labelled sample audio already committed to the "
                "Project Echo repository under models-and-data/samples/store_audio/. "
                "Provenance/collection method and external licence are not documented "
                "in the repository; treat as internal project reference audio only "
                "(see known_limitations.md)."
            ),
        }
        write_and_register(f"clean/{name}__clean.wav", audio, sr, row)

    # ---- 2. Background noise condition: source + pink noise at target SNR ----
    for name, info in SOURCE_SPECIES.items():
        audio, sr = clean_audio_cache[name]
        noise = pink_noise(len(audio), rng)
        mixed = mix_at_snr(audio, noise, BACKGROUND_NOISE_SNR_DB)
        idx = label_lookup(info["scientific_name"])
        row = {
            "file_id": f"{name}__background_noise",
            "species_scientific_name": info["scientific_name"],
            "species_common_name": info["common_name"],
            "in_model_class_list": idx is not None,
            "model_class_index": idx,
            "condition_category": "background_noise",
            "derivation": (
                f"clean source + synthetic pink noise, mixed at {BACKGROUND_NOISE_SNR_DB:.0f} dB SNR "
                f"(seed={RANDOM_SEED})"
            ),
            "source_repo_path": info["repo_path"],
            "source_repo": "https://github.com/DataBytes-Organisation/Project-Echo",
            "source_note": "Derived from the clean recording above; noise is synthetic, not field-recorded.",
        }
        write_and_register(f"background_noise/{name}__noise_snr{int(BACKGROUND_NOISE_SNR_DB)}db.wav", mixed, sr, row)

    # ---- 3. Weak signal condition: attenuated call + noise near 0 dB SNR ----
    for name, info in SOURCE_SPECIES.items():
        audio, sr = clean_audio_cache[name]
        attenuated = audio * db_to_amplitude(WEAK_SIGNAL_ATTENUATION_DB)
        noise = pink_noise(len(audio), rng)
        mixed = mix_at_snr(attenuated, noise, WEAK_SIGNAL_SNR_DB)
        idx = label_lookup(info["scientific_name"])
        row = {
            "file_id": f"{name}__weak_signal",
            "species_scientific_name": info["scientific_name"],
            "species_common_name": info["common_name"],
            "in_model_class_list": idx is not None,
            "model_class_index": idx,
            "condition_category": "weak_signal",
            "derivation": (
                f"clean source attenuated {WEAK_SIGNAL_ATTENUATION_DB:.0f} dB, then mixed with synthetic "
                f"pink noise at {WEAK_SIGNAL_SNR_DB:.0f} dB SNR (seed={RANDOM_SEED})"
            ),
            "source_repo_path": info["repo_path"],
            "source_repo": "https://github.com/DataBytes-Organisation/Project-Echo",
            "source_note": "Derived from the clean recording above; simulates a distant/faint call.",
        }
        write_and_register(f"weak_signal/{name}__weak_snr{int(WEAK_SIGNAL_SNR_DB)}db.wav", mixed, sr, row)

    # ---- 4. Overlapping calls condition: two species mixed together ----
    for a, b in OVERLAP_PAIRS:
        audio_a, sr_a = clean_audio_cache[a]
        audio_b, sr_b = clean_audio_cache[b]
        n = min(len(audio_a), len(audio_b))
        mix = mix_at_snr(audio_a[:n], audio_b[:n], OVERLAP_RELATIVE_LEVEL_DB)
        idx_a = label_lookup(SOURCE_SPECIES[a]["scientific_name"])
        idx_b = label_lookup(SOURCE_SPECIES[b]["scientific_name"])
        row = {
            "file_id": f"{a}_AND_{b}__overlapping",
            "species_scientific_name": f"{SOURCE_SPECIES[a]['scientific_name']} + {SOURCE_SPECIES[b]['scientific_name']}",
            "species_common_name": f"{SOURCE_SPECIES[a]['common_name']} + {SOURCE_SPECIES[b]['common_name']}",
            "in_model_class_list": bool(idx_a is not None and idx_b is not None),
            "model_class_index": f"{idx_a};{idx_b}",
            "condition_category": "overlapping_calls",
            "derivation": (
                f"two clean source recordings ({a}, {b}) mixed at "
                f"{OVERLAP_RELATIVE_LEVEL_DB:.0f} dB relative level, truncated to the shorter clip length"
            ),
            "source_repo_path": f"{SOURCE_SPECIES[a]['repo_path']} + {SOURCE_SPECIES[b]['repo_path']}",
            "source_repo": "https://github.com/DataBytes-Organisation/Project-Echo",
            "source_note": "Synthetic multi-species mixture; ground truth is a two-label overlap, not a single species.",
        }
        write_and_register(f"overlapping_calls/{a}_AND_{b}__overlapping.wav", mix, sr_a, row)

    # ---- Write metadata files ----
    fieldnames = list(metadata_rows[0].keys())
    # union of all keys across rows, keeping stable order
    for r in metadata_rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    csv_path = OUT_DIR / "evaluation_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in metadata_rows:
            writer.writerow(r)

    json_path = OUT_DIR / "evaluation_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_rows, f, indent=2)

    print(f"\nWrote {len(metadata_rows)} evaluation files.")
    print(f"Metadata: {csv_path}")
    print(f"Metadata: {json_path}")

    # ---- Preprocessing compatibility report ----
    n_pass = sum(1 for _, c in compat_rows if c["status"] == "PASS")
    n_fail = sum(1 for _, c in compat_rows if c["status"] == "FAIL")
    report_lines = []
    report_lines.append("# Engine Preprocessing Compatibility Report\n")
    report_lines.append(
        f"Generated by `scripts/build_evaluation_set.py`. "
        f"Every file below was passed through a faithful port of "
        f"`EchoEngine.efficientnetv2_preprocess_audio_bytes()` from "
        f"`src/production/engine/echo_engine.py` (the active production "
        f"preprocessing path, per `ACTIVE_INFERENCE_MODEL` in `echo_engine.json`).\n"
    )
    report_lines.append(f"**Preprocessing config used** (`models/efficientnetv2/preprocess_config.json`):\n")
    report_lines.append("```json\n" + json.dumps(preprocess_config, indent=2) + "\n```\n")
    report_lines.append(f"**Expected TFLite input tensor shape:** `{EXPECTED_TFLITE_INPUT_SHAPE}`\n")
    report_lines.append(f"\n## Summary\n")
    report_lines.append(f"- Files tested: {len(compat_rows)}\n")
    report_lines.append(f"- Passed: {n_pass}\n")
    report_lines.append(f"- Failed: {n_fail}\n")
    report_lines.append(f"\n## Per-file results\n")
    report_lines.append("| File | Status | Tensor shape | Matches expected shape | NaN/Inf | min | max | time (ms) | Error |\n")
    report_lines.append("|---|---|---|---|---|---|---|---|---|\n")
    for rel_path, c in compat_rows:
        report_lines.append(
            f"| {rel_path} | {c['status']} | {c['input_tensor_shape']} | "
            f"{c['matches_expected_tflite_shape']} | {c['contains_nan_or_inf']} | "
            f"{c['value_min']} | {c['value_max']} | {c['processing_time_ms']} | {c['error'] or ''} |\n"
        )

    report_lines.append(f"\n## Legacy `combined_pipeline` configuration (for reference only)\n")
    report_lines.append(
        "The Engine also contains an older `combined_pipeline()` preprocessing path "
        "configured via `echo_engine.json` (not `preprocess_config.json`). It uses a "
        "**different sample rate and mel-spectrogram configuration** to the active "
        "EfficientNetV2 TFLite model. This is a pre-existing integration gap in the "
        "repository (also independently observed by Krish's Sprint 1 integration-gap "
        "task), not something introduced by this evaluation set:\n\n"
    )
    report_lines.append("| Parameter | `preprocess_config.json` (active model) | `echo_engine.json` (`combined_pipeline`) |\n")
    report_lines.append("|---|---|---|\n")
    report_lines.append(f"| Sample rate | {preprocess_config['target_sr']} Hz | {legacy_config['AUDIO_SAMPLE_RATE']} Hz |\n")
    report_lines.append(f"| Clip duration | {preprocess_config['duration_s']} s | {legacy_config['AUDIO_CLIP_DURATION']} s |\n")
    report_lines.append(f"| Mel bands | {preprocess_config['n_mels']} | {legacy_config['AUDIO_MELS']} |\n")
    report_lines.append(f"| fmin / fmax | {preprocess_config['fmin']} / {preprocess_config['fmax']} Hz | {legacy_config['AUDIO_FMIN']} / {legacy_config['AUDIO_FMAX']} Hz |\n")
    report_lines.append(
        "\nRecommendation: confirm with Krish/Anand which configuration is authoritative "
        "before Sprint 2 field-calibration work (Dinal) reruns against this evaluation set.\n"
    )

    report_path = OUT_DIR / "preprocessing_compatibility_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
