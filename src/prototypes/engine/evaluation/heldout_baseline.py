"""Build and report the Sprint 2 held-out-only EfficientNetV2 baseline.

This module deliberately separates the portable shared manifest requested by
the calibration task from the richer Sprint 1 audit manifest. The shared CSV
has exactly four columns: ``filepath``, ``species``, ``label_id`` and ``split``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SHARED_COLUMNS = ["filepath", "species", "label_id", "split"]
HELDOUT_TIER = "reconstructed_heldout"
HELDOUT_SPLIT = "heldout"
EXPECTED_HELDOUT_ROWS = 737
DEFAULT_CLASS_MAPPING = (
    Path(__file__).resolve().parents[3]
    / "production"
    / "engine"
    / "models"
    / "efficientnetv2"
    / "class_mapping.json"
)


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.casefold()
    accepted = {"true", "false", "1", "0", "yes", "no", ""}
    unknown = normalized[~normalized.isin(accepted)]
    if not unknown.empty:
        raise ValueError(f"Unrecognised boolean values: {sorted(unknown.unique())}")
    return normalized.isin({"true", "1", "yes"})


def load_classes(class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING) -> list[str]:
    with Path(class_mapping_path).open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    classes = mapping.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError("Class mapping must contain a non-empty 'classes' list.")
    if len(classes) != len(set(classes)):
        raise ValueError("Class mapping contains duplicate species names.")
    return [str(value) for value in classes]


def validate_shared_manifest(
    manifest: pd.DataFrame,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING,
    expected_rows: int = EXPECTED_HELDOUT_ROWS,
) -> None:
    """Validate the shared manifest schema and its model-label alignment."""

    if list(manifest.columns) != SHARED_COLUMNS:
        raise ValueError(
            f"Shared manifest columns must be exactly {SHARED_COLUMNS}; "
            f"got {list(manifest.columns)}."
        )
    if len(manifest) != expected_rows:
        raise ValueError(
            f"Shared manifest must contain {expected_rows} rows; got {len(manifest)}."
        )
    if manifest["filepath"].astype(str).str.strip().eq("").any():
        raise ValueError("Shared manifest contains an empty filepath.")
    if manifest["filepath"].duplicated().any():
        duplicates = manifest.loc[
            manifest["filepath"].duplicated(keep=False), "filepath"
        ].tolist()
        raise ValueError(
            f"Shared manifest contains duplicate filepaths: {duplicates[:5]}"
        )
    if set(manifest["split"].astype(str)) != {HELDOUT_SPLIT}:
        raise ValueError(f"Every shared-manifest row must use split={HELDOUT_SPLIT!r}.")

    classes = load_classes(class_mapping_path)
    label_ids = pd.to_numeric(manifest["label_id"], errors="raise").astype(int)
    if ((label_ids < 0) | (label_ids >= len(classes))).any():
        raise ValueError(
            "Shared manifest contains a label_id outside the model mapping."
        )
    expected_species = label_ids.map(dict(enumerate(classes)))
    mismatched = (
        manifest["species"].astype(str).to_numpy() != expected_species.to_numpy()
    )
    if mismatched.any():
        first = int(np.flatnonzero(mismatched)[0])
        raise ValueError(
            "Species/label_id mismatch at row "
            f"{first}: {manifest.iloc[first]['species']!r} does not match "
            f"label_id {label_ids.iloc[first]}."
        )


def build_shared_manifest(
    source_manifest_path: str | Path,
    shared_output_path: str | Path,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING,
    detailed_output_path: str | Path | None = None,
    expected_rows: int = EXPECTED_HELDOUT_ROWS,
) -> pd.DataFrame:
    """Filter the Sprint 1 audit manifest to verified reconstructed held-out rows."""

    source = pd.read_csv(source_manifest_path, keep_default_na=False)
    required = {
        "relative_path",
        "true_label",
        "class_index",
        "selection_tier",
        "possible_training_overlap",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source manifest is missing columns: {sorted(missing)}")

    overlap = _boolean_series(source["possible_training_overlap"])
    heldout = source[(source["selection_tier"] == HELDOUT_TIER) & ~overlap].copy()
    heldout["class_index"] = pd.to_numeric(
        heldout["class_index"], errors="raise"
    ).astype(int)
    heldout = heldout.sort_values(
        ["class_index", "relative_path"], kind="stable"
    ).reset_index(drop=True)

    shared = pd.DataFrame(
        {
            "filepath": heldout["relative_path"].astype(str),
            "species": heldout["true_label"].astype(str),
            "label_id": heldout["class_index"],
            "split": HELDOUT_SPLIT,
        },
        columns=SHARED_COLUMNS,
    )
    validate_shared_manifest(shared, class_mapping_path, expected_rows)

    shared_output_path = Path(shared_output_path)
    shared_output_path.parent.mkdir(parents=True, exist_ok=True)
    shared.to_csv(shared_output_path, index=False, lineterminator="\n")

    if detailed_output_path is not None:
        detailed_output_path = Path(detailed_output_path)
        detailed_output_path.parent.mkdir(parents=True, exist_ok=True)
        heldout.to_csv(detailed_output_path, index=False, lineterminator="\n")
    return shared


def summarize_heldout_predictions(
    predictions_path: str | Path,
    shared_manifest_path: str | Path,
    output_dir: str | Path,
    class_mapping_path: str | Path = DEFAULT_CLASS_MAPPING,
    poor_recall_threshold: float = 0.9,
    expected_rows: int = EXPECTED_HELDOUT_ROWS,
) -> dict[str, object]:
    """Recompute held-out metrics and poor species from per-file predictions."""

    from sklearn.metrics import precision_recall_fscore_support

    shared = pd.read_csv(shared_manifest_path, keep_default_na=False)
    validate_shared_manifest(shared, class_mapping_path, expected_rows)
    predictions = pd.read_csv(predictions_path, keep_default_na=False)
    required = {
        "relative_path",
        "true_index",
        "true_label",
        "selection_tier",
        "possible_training_overlap",
        "predicted_index",
        "confidence",
        "top_3_correct",
        "top_5_correct",
        "correct",
        "inference_ms",
        "error",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing columns: {sorted(missing)}")
    if predictions["relative_path"].duplicated().any():
        raise ValueError("Predictions contain duplicate relative_path values.")

    selected = shared.merge(
        predictions,
        left_on="filepath",
        right_on="relative_path",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_predictions = selected[selected["_merge"] != "both"]["filepath"].tolist()
    if missing_predictions:
        raise ValueError(
            f"Predictions are missing shared-manifest files: {missing_predictions[:5]}"
        )
    selected = selected.drop(columns=["_merge"])

    if not selected["selection_tier"].eq(HELDOUT_TIER).all():
        raise ValueError("Selected predictions include a non-held-out selection tier.")
    if _boolean_series(selected["possible_training_overlap"]).any():
        raise ValueError("Selected predictions include possible training overlap.")
    if not selected["species"].eq(selected["true_label"]).all():
        raise ValueError("Prediction species do not match the shared manifest.")
    true_indices = pd.to_numeric(selected["true_index"], errors="raise").astype(int)
    label_ids = pd.to_numeric(selected["label_id"], errors="raise").astype(int)
    if not true_indices.equals(label_ids):
        raise ValueError("Prediction label IDs do not match the shared manifest.")

    successful = selected[selected["error"].eq("")].copy()
    if successful.empty:
        raise RuntimeError("Every held-out prediction failed.")
    classes = load_classes(class_mapping_path)
    y_true = pd.to_numeric(successful["true_index"], errors="raise").astype(int)
    y_pred = pd.to_numeric(successful["predicted_index"], errors="raise").astype(int)
    labels = np.arange(len(classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    correct = _boolean_series(successful["correct"])
    top_3 = _boolean_series(successful["top_3_correct"])
    top_5 = _boolean_series(successful["top_5_correct"])
    per_species = pd.DataFrame(
        {
            "label_id": labels,
            "species": classes,
            "support": support.astype(int),
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
    )
    poor_species = per_species[
        (per_species["support"] > 0) & (per_species["recall"] < poor_recall_threshold)
    ].sort_values(["recall", "f1_score", "species"], kind="stable")
    evaluated_mask = support > 0

    metrics: dict[str, object] = {
        "baseline_scope": "reconstructed_heldout_only",
        "manifest_rows": len(shared),
        "successful_files": len(successful),
        "failed_files": int(len(shared) - len(successful)),
        "model_classes": len(classes),
        "evaluated_species": int((support > 0).sum()),
        "top_1_accuracy": float(correct.mean()),
        "top_3_accuracy": float(top_3.mean()),
        "top_5_accuracy": float(top_5.mean()),
        "macro_precision_all_model_classes": float(np.mean(precision)),
        "macro_recall_all_model_classes": float(np.mean(recall)),
        "macro_f1_all_model_classes": float(np.mean(f1)),
        "macro_precision_evaluated_species": float(np.mean(precision[evaluated_mask])),
        "macro_recall_evaluated_species": float(np.mean(recall[evaluated_mask])),
        "macro_f1_evaluated_species": float(np.mean(f1[evaluated_mask])),
        "mean_confidence": float(
            pd.to_numeric(successful["confidence"], errors="raise").mean()
        ),
        "mean_inference_ms": float(
            pd.to_numeric(successful["inference_ms"], errors="raise").mean()
        ),
        "poor_recall_threshold": float(poor_recall_threshold),
        "poor_species_count": len(poor_species),
        "shared_manifest_sha256": sha256_file(shared_manifest_path),
        "source_predictions_sha256": sha256_file(predictions_path),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_species.to_csv(
        output_dir / "heldout_per_species_metrics.csv",
        index=False,
        lineterminator="\n",
    )
    poor_species.to_csv(
        output_dir / "heldout_poor_performing_species.csv",
        index=False,
        lineterminator="\n",
    )
    with (output_dir / "heldout_metrics.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")
    report_lines = [
        "# Held-Out Baseline Re-Evaluation",
        "",
        (
            "This report recomputes the rigorous held-out-only baseline from the "
            "verified Sprint 1 per-file predictions. It excludes all 405 "
            "unverified backfills."
        ),
        "",
        f"- Shared manifest rows: {metrics['manifest_rows']}",
        f"- Successful files: {metrics['successful_files']}",
        f"- Failed files: {metrics['failed_files']}",
        f"- Evaluated species: {metrics['evaluated_species']} / {metrics['model_classes']}",
        f"- Top-1 accuracy: {metrics['top_1_accuracy']:.2%}",
        f"- Top-3 accuracy: {metrics['top_3_accuracy']:.2%}",
        f"- Top-5 accuracy: {metrics['top_5_accuracy']:.2%}",
        (
            "- Macro precision (evaluated species): "
            f"{metrics['macro_precision_evaluated_species']:.2%}"
        ),
        (
            "- Macro recall (evaluated species): "
            f"{metrics['macro_recall_evaluated_species']:.2%}"
        ),
        f"- Macro F1 (evaluated species): {metrics['macro_f1_evaluated_species']:.2%}",
        (
            f"- Poor-performing species: {metrics['poor_species_count']} "
            f"(recall < {metrics['poor_recall_threshold']:.2f})"
        ),
        f"- Shared manifest SHA-256: `{metrics['shared_manifest_sha256']}`",
        "",
        (
            "A fresh inference run requires the fixed audio dataset and production "
            "model artifacts. The balanced-dataset rerun remains a separate Sprint 2 "
            "dependency."
        ),
        "",
    ]
    (output_dir / "heldout_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8", newline="\n"
    )
    return metrics


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-manifest", help="Create the shared held-out-only CSV"
    )
    build.add_argument("--source-manifest", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--detailed-output")
    build.add_argument("--class-mapping", default=str(DEFAULT_CLASS_MAPPING))
    build.add_argument("--expected-rows", type=int, default=EXPECTED_HELDOUT_ROWS)

    summarize = subparsers.add_parser(
        "summarize", help="Recompute held-out-only metrics from predictions"
    )
    summarize.add_argument("--predictions", required=True)
    summarize.add_argument("--manifest", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--class-mapping", default=str(DEFAULT_CLASS_MAPPING))
    summarize.add_argument("--expected-rows", type=int, default=EXPECTED_HELDOUT_ROWS)
    summarize.add_argument("--poor-recall-threshold", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "build-manifest":
        build_shared_manifest(
            args.source_manifest,
            args.output,
            args.class_mapping,
            args.detailed_output,
            args.expected_rows,
        )
    elif args.command == "summarize":
        summarize_heldout_predictions(
            args.predictions,
            args.manifest,
            args.output_dir,
            args.class_mapping,
            args.poor_recall_threshold,
            args.expected_rows,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
