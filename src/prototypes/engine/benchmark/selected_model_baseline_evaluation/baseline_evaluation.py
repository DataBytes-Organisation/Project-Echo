"""Reproducible balanced validation and TFLite evaluation for Project Echo.

The command-line workflow is intentionally split into auditable stages:

1. ``inventory`` lists object metadata from the configured GCS buckets.
2. ``select`` reconstructs the training notebook's held-out split and ranks
   fixed validation candidates.
3. ``download`` downloads only the selected candidates and reserves.
4. ``finalize`` decodes candidates and creates the fixed balanced dataset.
5. ``evaluate`` runs TFLite inference and produces all Sprint 1 reports.

No source objects are modified. The original model-training notebook did not
save its train/test manifest, so the held-out split is explicitly labelled a
reconstruction using its recorded settings (test_size=0.2, random_state=42).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("project_echo.baseline_evaluation")
DEFAULT_CONFIG = Path(__file__).with_name("evaluation_config.json")


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_label(value: str) -> str:
    """Normalise harmless label variations without changing taxonomy."""

    value = unicodedata.normalize("NFKC", str(value)).replace("_", " ")
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.casefold().split())


def stable_rank(seed: int, uri: str) -> str:
    """Return a stable rank key independent of filesystem enumeration order."""

    return hashlib.sha256(f"{seed}|{uri}".encode("utf-8")).hexdigest()


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def scores_to_probabilities(values: np.ndarray) -> tuple[np.ndarray, str]:
    """Use model probabilities directly, otherwise apply stable softmax."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        np.all(np.isfinite(values))
        and np.all(values >= -1e-6)
        and np.all(values <= 1.0 + 1e-6)
        and math.isclose(float(values.sum()), 1.0, rel_tol=1e-4, abs_tol=1e-4)
    ):
        clipped = np.clip(values, 0.0, 1.0)
        return clipped / clipped.sum(), "probabilities"
    return softmax(values), "logits_softmax"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def resolve_config_path(config_path: str | Path, configured_path: str) -> Path:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        return candidate
    return (Path(config_path).resolve().parent / candidate).resolve()


def load_class_mapping(path: str | Path) -> dict[str, Any]:
    mapping = load_json(path)
    required = {"classes", "label_to_index", "index_to_label"}
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"Class mapping is missing keys: {sorted(missing)}")
    classes = mapping["classes"]
    if len(classes) != len(set(classes)):
        raise ValueError("Class mapping contains duplicate class labels.")
    for index, label in enumerate(classes):
        if mapping["index_to_label"].get(str(index)) != label:
            raise ValueError(f"Inconsistent index_to_label entry at {index}.")
    return mapping


def run_gcloud_json(gcloud: str | Path, bucket: str) -> list[dict[str, Any]]:
    command = [
        str(gcloud),
        "storage",
        "ls",
        "--recursive",
        "--json",
        f"gs://{bucket}/",
    ]
    LOGGER.info("Listing object metadata from gs://%s/", bucket)
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return json.loads(completed.stdout)


def is_augmented(name: str, markers: Sequence[str]) -> bool:
    compact_name = re.sub(r"[^a-z0-9]", "", name.casefold())
    return any(
        re.sub(r"[^a-z0-9]", "", marker.casefold()) in compact_name
        for marker in markers
    )


def boolean_series(values: pd.Series) -> pd.Series:
    """Parse CSV boolean fields without treating the string ``False`` as true."""

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.casefold()
    unknown = normalized[~normalized.isin({"true", "false", "1", "0", "yes", "no", ""})]
    if not unknown.empty:
        raise ValueError(f"Unrecognised boolean values: {sorted(unknown.unique())}")
    return normalized.isin({"true", "1", "yes"})


def build_inventory(
    config_path: str | Path,
    gcloud: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """List candidate GCS audio metadata and map folders to model classes."""

    config = load_json(config_path)
    mapping_path = resolve_config_path(
        config_path, config["model"]["class_mapping_path"]
    )
    mapping = load_class_mapping(mapping_path)
    classes = mapping["classes"]
    canonical_to_label = {canonical_label(label): label for label in classes}
    valid_exts = {value.casefold() for value in config["selection"]["valid_extensions"]}
    markers = config["selection"]["augmentation_markers"]

    rows: list[dict[str, Any]] = []
    for source in config["dataset_sources"]:
        bucket = source["bucket"]
        for item in run_gcloud_json(gcloud, bucket):
            if item.get("type") != "cloud_object":
                continue
            metadata = item.get("metadata", {})
            object_name = metadata.get("name", "")
            if "/" not in object_name:
                continue
            source_label, _ = object_name.split("/", 1)
            model_label = canonical_to_label.get(canonical_label(source_label))
            suffix = Path(object_name).suffix.casefold()
            exclusion_reason = ""
            if model_label is None:
                exclusion_reason = "label_not_in_model_mapping"
            elif suffix not in valid_exts:
                exclusion_reason = "unsupported_extension"
            elif is_augmented(object_name, markers):
                exclusion_reason = "augmentation_marker"
            elif int(metadata.get("size", 0)) <= 0:
                exclusion_reason = "empty_object"

            generation = str(metadata.get("generation", ""))
            uri = f"gs://{bucket}/{object_name}"
            rows.append(
                {
                    "gcs_uri": uri,
                    "bucket": bucket,
                    "object_name": object_name,
                    "source_label": source_label,
                    "true_label": model_label or "",
                    "class_index": (
                        mapping["label_to_index"].get(model_label, "")
                        if model_label
                        else ""
                    ),
                    "generation": generation,
                    "size_bytes": int(metadata.get("size", 0)),
                    "md5_base64": metadata.get("md5Hash", ""),
                    "crc32c_base64": metadata.get("crc32c", ""),
                    "content_type": metadata.get("contentType", ""),
                    "extension": suffix,
                    "time_created": metadata.get("timeCreated", ""),
                    "excluded": bool(exclusion_reason),
                    "exclusion_reason": exclusion_reason,
                }
            )

    inventory = pd.DataFrame(rows).sort_values(
        ["class_index", "object_name", "bucket"], kind="stable", na_position="last"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_path, index=False, lineterminator="\n")

    included = inventory[~inventory["excluded"]]
    missing = sorted(set(classes) - set(included["true_label"]))
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_csv": str(output_path.resolve()),
        "inventory_sha256": sha256_file(output_path),
        "total_objects": int(len(inventory)),
        "eligible_audio_objects": int(len(included)),
        "eligible_classes": int(included["true_label"].nunique()),
        "expected_classes": len(classes),
        "missing_classes": missing,
        "source_counts": {
            str(key): int(value)
            for key, value in included.groupby("bucket").size().items()
        },
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    if missing:
        LOGGER.warning(
            "No compliant audio remains for %d model classes: %s",
            len(missing),
            missing,
        )
    return inventory


def reconstruct_split_and_select(
    config_path: str | Path,
    inventory_path: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct the documented held-out split and rank fixed candidates."""

    from sklearn.model_selection import train_test_split

    config = load_json(config_path)
    mapping_path = resolve_config_path(
        config_path, config["model"]["class_mapping_path"]
    )
    mapping = load_class_mapping(mapping_path)
    selection = config["selection"]
    inventory = pd.read_csv(inventory_path, keep_default_na=False)
    eligible = inventory[~boolean_series(inventory["excluded"])].copy()
    eligible["class_index"] = eligible["class_index"].astype(int)
    eligible = eligible.sort_values(
        ["class_index", "object_name", "bucket"], kind="stable"
    ).reset_index(drop=True)

    duplicate_key = eligible["md5_base64"].where(
        eligible["md5_base64"].astype(bool), eligible["gcs_uri"]
    )
    eligible["duplicate_key"] = duplicate_key
    eligible["exact_duplicate"] = eligible.duplicated(
        subset=["true_label", "duplicate_key"], keep="first"
    )
    deduplicated = eligible[~eligible["exact_duplicate"]].copy().reset_index(drop=True)

    counts = deduplicated.groupby("class_index").size()
    if not counts.empty and int(counts.min()) < 2:
        labels = counts[counts < 2].index.tolist()
        raise RuntimeError(f"Classes with fewer than two files: {labels}")

    all_indices = np.arange(len(deduplicated))
    labels = deduplicated["class_index"].to_numpy()
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=float(selection["original_test_size"]),
        random_state=int(selection["original_split_random_state"]),
        stratify=labels,
    )
    split = np.full(len(deduplicated), "train", dtype=object)
    split[validation_indices] = "validation"
    deduplicated["reconstructed_split"] = split
    deduplicated["split_method"] = (
        "reconstructed_sklearn_stratified_test_split_unverified"
    )

    candidate_pool = deduplicated.copy()
    seed = int(selection["selection_seed"])
    candidate_pool["stable_rank_key"] = candidate_pool["gcs_uri"].map(
        lambda uri: stable_rank(seed, uri)
    )
    candidate_pool["selection_tier"] = np.where(
        candidate_pool["reconstructed_split"] == "validation",
        "reconstructed_heldout",
        "unverified_backfill",
    )
    candidate_pool["possible_training_overlap"] = (
        candidate_pool["selection_tier"] == "unverified_backfill"
    )
    candidate_pool["tier_priority"] = candidate_pool[
        "possible_training_overlap"
    ].astype(int)
    candidate_pool = candidate_pool.sort_values(
        ["class_index", "tier_priority", "stable_rank_key"], kind="stable"
    )
    candidate_pool["candidate_rank"] = (
        candidate_pool.groupby("class_index").cumcount() + 1
    )
    target = int(selection["files_per_species"])
    reserves = int(selection["reserve_files_per_species"])
    candidate_pool["primary_candidate"] = candidate_pool["candidate_rank"] <= target
    candidate_pool["download_candidate"] = (
        candidate_pool["candidate_rank"] <= target + reserves
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "selection_audit.csv"
    candidate_pool.to_csv(audit_path, index=False, lineterminator="\n")

    eligible_counts = candidate_pool.groupby("true_label").size()
    heldout_counts = candidate_pool[
        candidate_pool["selection_tier"] == "reconstructed_heldout"
    ].groupby("true_label").size()
    coverage_rows = []
    for class_index, label in enumerate(mapping["classes"]):
        available = int(eligible_counts.get(label, 0))
        heldout = int(heldout_counts.get(label, 0))
        selected = min(target, available)
        coverage_rows.append(
            {
                "class_index": class_index,
                "true_label": label,
                "eligible_deduplicated_files": available,
                "reconstructed_heldout_files": heldout,
                "target_files": target,
                "planned_selected_files": selected,
                "planned_unverified_backfill_files": max(0, selected - heldout),
                "shortage_before_decode": max(0, target - selected),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "selection_coverage.csv", index=False, lineterminator="\n")
    write_json(
        output_dir / "selection_metadata.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inventory_sha256": sha256_file(inventory_path),
            "selection_audit_sha256": sha256_file(audit_path),
            "ordering": "class_index, object_name, bucket",
            "original_test_size": float(selection["original_test_size"]),
            "original_split_random_state": int(
                selection["original_split_random_state"]
            ),
            "selection_seed": seed,
            "target_files_per_species": target,
            "reserve_files_per_species": reserves,
            "split_warning": (
                "The original training notebook did not save its split manifest. "
                "This held-out split is a deterministic reconstruction from the "
                "recorded settings and cannot prove exact identity with the original. "
                "Reconstructed held-out candidates are selected first. Remaining "
                "shortages are backfilled from the eligible pool and explicitly marked "
                "as possible training overlap."
            ),
        },
    )
    return candidate_pool, coverage


def safe_component(value: str) -> str:
    value = re.sub(r"[<>:\"/\\|?*]", "_", value).strip(" .")
    return value or "unknown"


def download_candidates(
    selection_audit_path: str | Path,
    gcloud: str | Path,
    cache_dir: str | Path,
) -> pd.DataFrame:
    """Download primary and reserve candidates without modifying GCS."""

    audit = pd.read_csv(selection_audit_path, keep_default_na=False)
    candidates = audit[boolean_series(audit["download_candidate"])].copy()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths: dict[str, str] = {}
    download_groups: list[tuple[str, Path, list[str]]] = []

    for label, group in candidates.groupby("true_label", sort=False):
        destination = cache_dir / safe_component(label)
        destination.mkdir(parents=True, exist_ok=True)
        missing_uris = []
        for row in group.itertuples(index=False):
            local_path = destination / Path(row.object_name).name
            local_paths[row.gcs_uri] = str(local_path.resolve())
            if not local_path.exists():
                missing_uris.append(row.gcs_uri)
        if missing_uris:
            download_groups.append((label, destination, missing_uris))

    def download_group(task: tuple[str, Path, list[str]]) -> None:
        label, destination, missing_uris = task
        LOGGER.info("Downloading %d candidates for %s", len(missing_uris), label)
        subprocess.run(
            [str(gcloud), "storage", "cp", "--quiet", *missing_uris, str(destination)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    if download_groups:
        workers = min(8, len(download_groups))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(download_group, download_groups))

    candidates["cache_path"] = candidates["gcs_uri"].map(local_paths)
    candidates["downloaded"] = candidates["cache_path"].map(
        lambda value: Path(value).is_file()
    )
    output_path = Path(selection_audit_path).with_name("downloaded_candidates.csv")
    candidates.to_csv(output_path, index=False, lineterminator="\n")
    return candidates


def validate_audio_file(path: Path, target_sr: int, duration_s: float) -> tuple[bool, str, float]:
    import librosa

    try:
        audio, sample_rate = librosa.load(
            str(path), sr=target_sr, mono=True, duration=duration_s
        )
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            return False, "empty_or_non_finite_audio", 0.0
        return True, "", float(audio.size / sample_rate)
    except Exception as error:  # noqa: BLE001 - error must be reported per file
        return False, f"{type(error).__name__}: {error}", 0.0


def finalize_dataset(
    config_path: str | Path,
    downloaded_candidates_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> pd.DataFrame:
    """Select the first readable candidates and copy the fixed dataset."""

    config = load_json(config_path)
    preprocess_path = resolve_config_path(
        config_path, config["model"]["preprocess_config_path"]
    )
    preprocess = load_json(preprocess_path)
    target = int(config["selection"]["files_per_species"])
    mapping_path = resolve_config_path(
        config_path, config["model"]["class_mapping_path"]
    )
    mapping = load_class_mapping(mapping_path)
    candidates = pd.read_csv(downloaded_candidates_path, keep_default_na=False)
    candidates = candidates.sort_values(["class_index", "candidate_rank"], kind="stable")
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for class_index, label in enumerate(mapping["classes"]):
        group = candidates[candidates["true_label"] == label]
        selected = 0
        for row in group.itertuples(index=False):
            source_path = Path(row.cache_path)
            readable, error, decoded_seconds = validate_audio_file(
                source_path,
                int(preprocess["target_sr"]),
                float(preprocess["duration_s"]),
            )
            validation_rows.append(
                {
                    "class_index": class_index,
                    "true_label": label,
                    "candidate_rank": int(row.candidate_rank),
                    "gcs_uri": row.gcs_uri,
                    "cache_file": source_path.name,
                    "readable": readable,
                    "decoded_seconds_checked": decoded_seconds,
                    "validation_error": error,
                }
            )
            if not readable or selected >= target:
                continue
            selected += 1
            label_dir = dataset_dir / safe_component(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            destination = label_dir / source_path.name
            source_hash = sha256_file(source_path)
            if destination.exists() and sha256_file(destination) != source_hash:
                raise FileExistsError(
                    f"Refusing to overwrite different dataset file: {destination}"
                )
            if not destination.exists():
                shutil.copy2(source_path, destination)
            manifest_rows.append(
                {
                    "manifest_version": 1,
                    "class_index": class_index,
                    "true_label": label,
                    "selection_rank": selected,
                    "gcs_uri": row.gcs_uri,
                    "generation": row.generation,
                    "size_bytes": int(row.size_bytes),
                    "gcs_md5_base64": row.md5_base64,
                    "gcs_crc32c_base64": row.crc32c_base64,
                    "sha256": source_hash,
                    "relative_path": destination.relative_to(dataset_dir).as_posix(),
                    "split_method": row.split_method,
                    "selection_tier": row.selection_tier,
                    "possible_training_overlap": bool(row.possible_training_overlap),
                }
            )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["class_index", "selection_rank"], kind="stable"
    )
    manifest_path = output_dir / "balanced_validation_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    pd.DataFrame(validation_rows).to_csv(
        output_dir / "candidate_decode_validation.csv", index=False, lineterminator="\n"
    )

    counts = manifest.groupby("true_label").size()
    shortages = []
    for class_index, label in enumerate(mapping["classes"]):
        count = int(counts.get(label, 0))
        if count < target:
            shortages.append(
                {
                    "class_index": class_index,
                    "true_label": label,
                    "selected_files": count,
                    "target_files": target,
                    "shortage": target - count,
                }
            )
    pd.DataFrame(
        shortages,
        columns=[
            "class_index",
            "true_label",
            "selected_files",
            "target_files",
            "shortage",
        ],
    ).to_csv(output_dir / "dataset_shortages.csv", index=False, lineterminator="\n")
    write_json(
        output_dir / "dataset_version.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(manifest_path),
            "preprocess_config_sha256": sha256_file(preprocess_path),
            "class_mapping_sha256": sha256_file(mapping_path),
            "selected_files": int(len(manifest)),
            "covered_classes": int(manifest["true_label"].nunique()),
            "expected_classes": len(mapping["classes"]),
            "shortage_classes": len(shortages),
            "reconstructed_heldout_files": int(
                (manifest["selection_tier"] == "reconstructed_heldout").sum()
            ),
            "unverified_backfill_files": int(
                (manifest["selection_tier"] == "unverified_backfill").sum()
            ),
        },
    )
    return manifest


def preprocess_audio(audio_path: Path, config: dict[str, Any]) -> np.ndarray:
    import librosa

    audio, sample_rate = librosa.load(
        str(audio_path), sr=int(config["target_sr"]), mono=True
    )
    target_length = int(float(config["duration_s"]) * int(config["target_sr"]))
    if audio.size < target_length:
        audio = np.pad(audio, (0, target_length - audio.size))
    else:
        audio = audio[:target_length]
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=int(config["n_mels"]),
        hop_length=int(config["hop_length"]),
        fmin=float(config["fmin"]),
        fmax=float(config["fmax"]),
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-6))[None, None, :, :]


def adapt_input(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    expected = tuple(int(value) for value in detail["shape"])
    candidate = array
    if tuple(candidate.shape) != expected:
        nhwc = np.transpose(array, (0, 2, 3, 1))
        if tuple(nhwc.shape) != expected:
            raise ValueError(
                f"Cannot adapt NCHW {array.shape} or NHWC {nhwc.shape} to {expected}."
            )
        candidate = nhwc
    dtype = detail["dtype"]
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if np.issubdtype(dtype, np.integer):
        if not scale:
            raise ValueError("Quantized input tensor has no scale.")
        limits = np.iinfo(dtype)
        candidate = np.clip(np.rint(candidate / scale + zero_point), limits.min, limits.max)
    return candidate.astype(dtype)


def dequantize_output(array: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    scale, zero_point = detail.get("quantization", (0.0, 0))
    if np.issubdtype(array.dtype, np.integer) and scale:
        return (array.astype(np.float32) - zero_point) * scale
    return array.astype(np.float32)


def create_interpreter(model_path: Path):
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(
            f"Expected one input and one output tensor; got {len(inputs)} and {len(outputs)}."
        )
    return interpreter, inputs[0], outputs[0]


@dataclass
class EvaluationArtifacts:
    predictions: Path
    metrics: Path
    per_species: Path
    confusion_matrix_csv: Path
    confusion_matrix_png: Path
    poor_species: Path
    report: Path


def evaluate_manifest(
    config_path: str | Path,
    manifest_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> EvaluationArtifacts:
    """Run TFLite inference and generate all baseline-evaluation evidence."""

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    config = load_json(config_path)
    model_path = resolve_config_path(config_path, config["model"]["tflite_path"])
    mapping_path = resolve_config_path(
        config_path, config["model"]["class_mapping_path"]
    )
    preprocess_path = resolve_config_path(
        config_path, config["model"]["preprocess_config_path"]
    )
    mapping = load_class_mapping(mapping_path)
    preprocess = load_json(preprocess_path)
    classes = mapping["classes"]
    label_to_index = {label: index for index, label in enumerate(classes)}
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interpreter, input_detail, output_detail = create_interpreter(model_path)
    output_shape = tuple(int(value) for value in output_detail["shape"])
    if output_shape[-1] != len(classes):
        raise ValueError(
            f"Model returns {output_shape[-1]} classes but mapping contains {len(classes)}."
        )

    result_rows: list[dict[str, Any]] = []
    probability_mode = ""
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        path = dataset_dir / row.relative_path
        started = time.perf_counter()
        try:
            features = preprocess_audio(path, preprocess)
            tflite_input = adapt_input(features, input_detail)
            inference_started = time.perf_counter()
            interpreter.set_tensor(input_detail["index"], tflite_input)
            interpreter.invoke()
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            raw = interpreter.get_tensor(output_detail["index"])[0]
            raw = dequantize_output(raw, output_detail)
            probabilities, probability_mode = scores_to_probabilities(raw)
            top_indices = np.argsort(probabilities)[::-1][:5]
            true_index = label_to_index[row.true_label]
            result = {
                "gcs_uri": row.gcs_uri,
                "relative_path": row.relative_path,
                "true_index": true_index,
                "true_label": row.true_label,
                "selection_tier": row.selection_tier,
                "possible_training_overlap": bool(row.possible_training_overlap),
                "predicted_index": int(top_indices[0]),
                "predicted_label": classes[int(top_indices[0])],
                "confidence": float(probabilities[top_indices[0]]),
                "top_3_correct": bool(true_index in top_indices[:3]),
                "top_5_correct": bool(true_index in top_indices[:5]),
                "correct": bool(true_index == top_indices[0]),
                "top_predictions": json.dumps(
                    [
                        {
                            "rank": rank,
                            "index": int(index),
                            "label": classes[int(index)],
                            "confidence": float(probabilities[index]),
                        }
                        for rank, index in enumerate(top_indices, start=1)
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "inference_ms": inference_ms,
                "total_processing_ms": (time.perf_counter() - started) * 1000.0,
                "error": "",
            }
        except Exception as error:  # noqa: BLE001 - report every failed file
            result = {
                "gcs_uri": row.gcs_uri,
                "relative_path": row.relative_path,
                "true_index": label_to_index.get(row.true_label, ""),
                "true_label": row.true_label,
                "selection_tier": row.selection_tier,
                "possible_training_overlap": bool(row.possible_training_overlap),
                "predicted_index": "",
                "predicted_label": "",
                "confidence": "",
                "top_3_correct": False,
                "top_5_correct": False,
                "correct": False,
                "top_predictions": "[]",
                "inference_ms": "",
                "total_processing_ms": (time.perf_counter() - started) * 1000.0,
                "error": f"{type(error).__name__}: {error}",
            }
        result_rows.append(result)
        if number % 50 == 0 or number == len(manifest):
            LOGGER.info("Evaluated %d/%d files", number, len(manifest))

    results = pd.DataFrame(result_rows)
    predictions_path = output_dir / "per_file_predictions.csv"
    results.to_csv(predictions_path, index=False, lineterminator="\n")
    successful = results[results["error"] == ""].copy()
    if successful.empty:
        raise RuntimeError("Every inference failed; see per_file_predictions.csv.")
    y_true = successful["true_index"].astype(int).to_numpy()
    y_pred = successful["predicted_index"].astype(int).to_numpy()
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_species_rows = []
    for index, label in enumerate(classes):
        species_rows = successful[successful["true_index"].astype(int) == index]
        failed_count = int(
            len(results[(results["true_label"] == label) & (results["error"] != "")])
        )
        per_species_rows.append(
            {
                "class_index": index,
                "species": label,
                "support": int(support[index]),
                "failed_files": failed_count,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "top_1_accuracy": float(species_rows["correct"].mean())
                if len(species_rows)
                else 0.0,
                "top_3_accuracy": float(species_rows["top_3_correct"].mean())
                if len(species_rows)
                else 0.0,
                "top_5_accuracy": float(species_rows["top_5_correct"].mean())
                if len(species_rows)
                else 0.0,
                "mean_confidence": float(species_rows["confidence"].astype(float).mean())
                if len(species_rows)
                else 0.0,
                "mean_inference_ms": float(
                    species_rows["inference_ms"].astype(float).mean()
                )
                if len(species_rows)
                else 0.0,
            }
        )
    per_species = pd.DataFrame(per_species_rows)
    per_species_path = output_dir / "per_species_metrics.csv"
    per_species.to_csv(per_species_path, index=False, lineterminator="\n")

    threshold = float(config["reporting"]["poor_species_recall_threshold"])
    poor = per_species[
        (per_species["support"] > 0) & (per_species["recall"] < threshold)
    ].sort_values(["recall", "f1_score", "species"], kind="stable")
    poor_path = output_dir / "poor_performing_species.csv"
    poor.to_csv(poor_path, index=False, lineterminator="\n")
    unevaluated = per_species[per_species["support"] == 0].copy()
    unevaluated_path = output_dir / "unevaluated_species.csv"
    unevaluated.to_csv(unevaluated_path, index=False, lineterminator="\n")

    tier_rows = []
    for tier, tier_data in successful.groupby("selection_tier", sort=False):
        tier_rows.append(
            {
                "selection_tier": tier,
                "files": int(len(tier_data)),
                "top_1_accuracy": float(tier_data["correct"].mean()),
                "top_3_accuracy": float(tier_data["top_3_correct"].mean()),
                "top_5_accuracy": float(tier_data["top_5_correct"].mean()),
                "mean_confidence": float(tier_data["confidence"].astype(float).mean()),
                "mean_inference_ms": float(
                    tier_data["inference_ms"].astype(float).mean()
                ),
            }
        )
    metrics_by_tier = pd.DataFrame(tier_rows)
    metrics_by_tier_path = output_dir / "metrics_by_selection_tier.csv"
    metrics_by_tier.to_csv(metrics_by_tier_path, index=False, lineterminator="\n")

    def tier_metric(tier: str, column: str) -> float | None:
        row = metrics_by_tier[metrics_by_tier["selection_tier"] == tier]
        return None if row.empty else float(row.iloc[0][column])

    matrix_frame = pd.DataFrame(matrix, index=classes, columns=classes)
    matrix_path = output_dir / "confusion_matrix.csv"
    matrix_frame.to_csv(matrix_path, lineterminator="\n")
    matrix_png = output_dir / "confusion_matrix.png"
    _plot_confusion_matrix(matrix, classes, matrix_png)

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    metrics = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": "EfficientNetV2 TFLite",
        "selected_files": int(len(results)),
        "successful_files": int(len(successful)),
        "failed_files": int(len(results) - len(successful)),
        "classes_in_model": len(classes),
        "classes_with_successful_files": int(successful["true_label"].nunique()),
        "reconstructed_heldout_files": int(
            (manifest["selection_tier"] == "reconstructed_heldout").sum()
        ),
        "unverified_backfill_files": int(
            (manifest["selection_tier"] == "unverified_backfill").sum()
        ),
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "top_3_accuracy": float(successful["top_3_correct"].mean()),
        "top_5_accuracy": float(successful["top_5_correct"].mean()),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "average_inference_ms": float(successful["inference_ms"].astype(float).mean()),
        "median_inference_ms": float(successful["inference_ms"].astype(float).median()),
        "probability_interpretation": probability_mode,
        "poor_species_recall_threshold": threshold,
        "poor_species_count": int(len(poor)),
        "unevaluated_species_count": int(len(unevaluated)),
        "model_sha256": sha256_file(model_path),
        "class_mapping_sha256": sha256_file(mapping_path),
        "preprocess_config_sha256": sha256_file(preprocess_path),
        "manifest_sha256": sha256_file(manifest_path),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "reconstructed_heldout_accuracy": tier_metric(
            "reconstructed_heldout", "top_1_accuracy"
        ),
        "reconstructed_heldout_top_3_accuracy": tier_metric(
            "reconstructed_heldout", "top_3_accuracy"
        ),
        "reconstructed_heldout_top_5_accuracy": tier_metric(
            "reconstructed_heldout", "top_5_accuracy"
        ),
        "unverified_backfill_accuracy": tier_metric(
            "unverified_backfill", "top_1_accuracy"
        ),
    }
    metrics_path = output_dir / "metrics_summary.json"
    write_json(metrics_path, metrics)
    report_path = output_dir / "evaluation_report.md"
    _write_report(report_path, metrics, poor, input_detail, output_detail)
    write_json(
        output_dir / "reproducibility.json",
        {
            **metrics,
            "config_path": Path(config_path).as_posix(),
            "config_sha256": sha256_file(config_path),
            "predictions_sha256": sha256_file(predictions_path),
            "per_species_sha256": sha256_file(per_species_path),
            "confusion_matrix_sha256": sha256_file(matrix_path),
            "poor_species_sha256": sha256_file(poor_path),
            "unevaluated_species_sha256": sha256_file(unevaluated_path),
            "metrics_by_selection_tier_sha256": sha256_file(metrics_by_tier_path),
            "tensorflow_version": _module_version("tensorflow"),
            "librosa_version": _module_version("librosa"),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
    )
    return EvaluationArtifacts(
        predictions=predictions_path,
        metrics=metrics_path,
        per_species=per_species_path,
        confusion_matrix_csv=matrix_path,
        confusion_matrix_png=matrix_png,
        poor_species=poor_path,
        report=report_path,
    )


def _module_version(name: str) -> str:
    module = __import__(name)
    return str(getattr(module, "__version__", "unknown"))


def _plot_confusion_matrix(matrix: np.ndarray, classes: Sequence[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(36, 30), dpi=180)
    sns.heatmap(
        normalized,
        cmap="Blues",
        vmin=0,
        vmax=1,
        xticklabels=classes,
        yticklabels=classes,
        square=True,
        cbar_kws={"label": "Recall-normalised proportion"},
        ax=axis,
    )
    axis.set_title("EfficientNetV2 TFLite - normalised confusion matrix")
    axis.set_xlabel("Predicted species")
    axis.set_ylabel("True species")
    axis.tick_params(axis="x", labelrotation=90, labelsize=5)
    axis.tick_params(axis="y", labelrotation=0, labelsize=5)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    metrics: dict[str, Any],
    poor: pd.DataFrame,
    input_detail: dict[str, Any],
    output_detail: dict[str, Any],
) -> None:
    poor_preview = poor.head(20)
    lines = [
        "# Selected-Model Baseline Evaluation",
        "",
        "## Summary",
        "",
        f"- Successfully evaluated files: {metrics['successful_files']} / {metrics['selected_files']}",
        f"- Reconstructed held-out files: {metrics['reconstructed_heldout_files']}",
        f"- Unverified backfill files: {metrics['unverified_backfill_files']}",
        f"- Reconstructed held-out top-1 accuracy: {metrics['reconstructed_heldout_accuracy']:.4f}",
        f"- Reconstructed held-out top-3 accuracy: {metrics['reconstructed_heldout_top_3_accuracy']:.4f}",
        f"- Reconstructed held-out top-5 accuracy: {metrics['reconstructed_heldout_top_5_accuracy']:.4f}",
        f"- Unverified backfill top-1 accuracy: {metrics['unverified_backfill_accuracy']:.4f}",
        f"- Overall accuracy: {metrics['overall_accuracy']:.4f}",
        f"- Top-3 accuracy: {metrics['top_3_accuracy']:.4f}",
        f"- Top-5 accuracy: {metrics['top_5_accuracy']:.4f}",
        f"- Macro precision: {metrics['macro_precision']:.4f}",
        f"- Macro recall: {metrics['macro_recall']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Average TFLite inference time: {metrics['average_inference_ms']:.2f} ms",
        f"- Poor-performing evaluated species (recall < {metrics['poor_species_recall_threshold']:.2f}): {metrics['poor_species_count']}",
        f"- Unevaluated species with no compliant files: {metrics['unevaluated_species_count']}",
        "",
        "## Model interface",
        "",
        f"- Input shape: {list(map(int, input_detail['shape']))}",
        f"- Input dtype: {np.dtype(input_detail['dtype']).name}",
        f"- Output shape: {list(map(int, output_detail['shape']))}",
        f"- Output dtype: {np.dtype(output_detail['dtype']).name}",
        f"- Output interpretation: {metrics['probability_interpretation']}",
        "",
        "## Lowest-performing species",
        "",
        "| Species | Support | Precision | Recall | F1 | Top-3 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in poor_preview.itertuples(index=False):
        lines.append(
            f"| {row.species} | {row.support} | {row.precision:.3f} | "
            f"{row.recall:.3f} | {row.f1_score:.3f} | "
            f"{row.top_3_accuracy:.3f} | {row.top_5_accuracy:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- The original training notebook recorded an 80/20 stratified split with random_state=42 but did not save the split manifest. Reconstructed held-out files are selected first. To meet the 10-per-species requirement where possible, shortages are filled from the remaining eligible pool and flagged as `unverified_backfill`; those rows may overlap training and must not be presented as an independent test set.",
            "- Many legacy segmented objects use names such as `region_start-end` and do not preserve a source-recording identifier. Exact duplicates are excluded using GCS MD5 metadata, but independence between clips from the same original recording cannot always be verified.",
            "- Metrics are calculated only from successfully decoded and inferred files; failures are listed in `per_file_predictions.csv`.",
            "",
            "## Evidence files",
            "",
            "- `metrics_summary.json`",
            "- `per_file_predictions.csv`",
            "- `per_species_metrics.csv`",
            "- `confusion_matrix.csv` and `confusion_matrix.png`",
            "- `poor_performing_species.csv`",
            "- `unevaluated_species.csv`",
            "- `metrics_by_selection_tier.csv`",
            "- `reproducibility.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="Inventory configured GCS sources")
    inventory.add_argument("--gcloud", required=True)
    inventory.add_argument("--output", required=True)

    select = subparsers.add_parser("select", help="Reconstruct held-out split and rank candidates")
    select.add_argument("--inventory", required=True)
    select.add_argument("--output-dir", required=True)

    download = subparsers.add_parser("download", help="Download primary and reserve candidates")
    download.add_argument("--selection-audit", required=True)
    download.add_argument("--gcloud", required=True)
    download.add_argument("--cache-dir", required=True)

    finalize = subparsers.add_parser("finalize", help="Decode candidates and create fixed dataset")
    finalize.add_argument("--downloaded-candidates", required=True)
    finalize.add_argument("--dataset-dir", required=True)
    finalize.add_argument("--output-dir", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Run TFLite evaluation and reports")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--dataset-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.command == "inventory":
        build_inventory(args.config, args.gcloud, args.output)
    elif args.command == "select":
        reconstruct_split_and_select(args.config, args.inventory, args.output_dir)
    elif args.command == "download":
        download_candidates(args.selection_audit, args.gcloud, args.cache_dir)
    elif args.command == "finalize":
        finalize_dataset(
            args.config,
            args.downloaded_candidates,
            args.dataset_dir,
            args.output_dir,
        )
    elif args.command == "evaluate":
        evaluate_manifest(args.config, args.manifest, args.dataset_dir, args.output_dir)
    else:  # pragma: no cover - argparse enforces the command
        parser.error(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
