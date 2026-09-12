from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "prototypes"
    / "engine"
    / "evaluation"
    / "heldout_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("heldout_baseline", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _class_mapping(path: Path) -> Path:
    path.write_text(
        json.dumps({"classes": ["Species A", "Species B"]}), encoding="utf-8"
    )
    return path


def test_build_shared_manifest_filters_backfills_and_writes_exact_schema(tmp_path):
    source = pd.DataFrame(
        [
            {
                "relative_path": "Species B/b.wav",
                "true_label": "Species B",
                "class_index": 1,
                "selection_tier": "reconstructed_heldout",
                "possible_training_overlap": False,
            },
            {
                "relative_path": "Species A/backfill.wav",
                "true_label": "Species A",
                "class_index": 0,
                "selection_tier": "unverified_backfill",
                "possible_training_overlap": True,
            },
            {
                "relative_path": "Species A/a.wav",
                "true_label": "Species A",
                "class_index": 0,
                "selection_tier": "reconstructed_heldout",
                "possible_training_overlap": False,
            },
        ]
    )
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "heldout.csv"
    source.to_csv(source_path, index=False)

    result = MODULE.build_shared_manifest(
        source_path,
        output_path,
        _class_mapping(tmp_path / "mapping.json"),
        expected_rows=2,
    )

    assert list(result.columns) == ["filepath", "species", "label_id", "split"]
    assert result["filepath"].tolist() == ["Species A/a.wav", "Species B/b.wav"]
    assert result["split"].tolist() == ["heldout", "heldout"]
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == (
        "filepath,species,label_id,split"
    )
    assert "Unnamed: 0" not in pd.read_csv(output_path).columns


def test_validate_shared_manifest_rejects_species_label_mismatch(tmp_path):
    manifest = pd.DataFrame(
        [
            {
                "filepath": "a.wav",
                "species": "Species B",
                "label_id": 0,
                "split": "heldout",
            }
        ]
    )
    with pytest.raises(ValueError, match="Species/label_id mismatch"):
        MODULE.validate_shared_manifest(
            manifest,
            _class_mapping(tmp_path / "mapping.json"),
            expected_rows=1,
        )


def test_summarize_heldout_predictions_uses_only_shared_rows(tmp_path):
    mapping_path = _class_mapping(tmp_path / "mapping.json")
    manifest = pd.DataFrame(
        [
            {
                "filepath": "Species A/a.wav",
                "species": "Species A",
                "label_id": 0,
                "split": "heldout",
            },
            {
                "filepath": "Species B/b.wav",
                "species": "Species B",
                "label_id": 1,
                "split": "heldout",
            },
        ]
    )
    manifest_path = tmp_path / "heldout.csv"
    manifest.to_csv(manifest_path, index=False)
    predictions = pd.DataFrame(
        [
            {
                "relative_path": "Species A/a.wav",
                "true_index": 0,
                "true_label": "Species A",
                "selection_tier": "reconstructed_heldout",
                "possible_training_overlap": False,
                "predicted_index": 0,
                "confidence": 0.9,
                "top_3_correct": True,
                "top_5_correct": True,
                "correct": True,
                "inference_ms": 10.0,
                "error": "",
            },
            {
                "relative_path": "Species B/b.wav",
                "true_index": 1,
                "true_label": "Species B",
                "selection_tier": "reconstructed_heldout",
                "possible_training_overlap": False,
                "predicted_index": 0,
                "confidence": 0.7,
                "top_3_correct": True,
                "top_5_correct": True,
                "correct": False,
                "inference_ms": 12.0,
                "error": "",
            },
            {
                "relative_path": "Species A/backfill.wav",
                "true_index": 0,
                "true_label": "Species A",
                "selection_tier": "unverified_backfill",
                "possible_training_overlap": True,
                "predicted_index": 0,
                "confidence": 1.0,
                "top_3_correct": True,
                "top_5_correct": True,
                "correct": True,
                "inference_ms": 1.0,
                "error": "",
            },
        ]
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    metrics = MODULE.summarize_heldout_predictions(
        predictions_path,
        manifest_path,
        tmp_path / "results",
        mapping_path,
        expected_rows=2,
    )

    assert metrics["manifest_rows"] == 2
    assert metrics["top_1_accuracy"] == 0.5
    assert metrics["top_3_accuracy"] == 1.0
    assert metrics["failed_files"] == 0
    assert (tmp_path / "results" / "heldout_report.md").is_file()
    poor = pd.read_csv(tmp_path / "results" / "heldout_poor_performing_species.csv")
    assert poor["species"].tolist() == ["Species B"]
