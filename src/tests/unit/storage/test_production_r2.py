import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def r2_environment(monkeypatch):
    monkeypatch.setenv("R2_ACCOUNT_ID", "account")
    monkeypatch.setenv("R2_BUCKET_NAME", "bucket")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_DATASET_PREFIX", "dataset")


def test_engine_lists_sorted_species_and_ignores_folder_markers(r2_environment):
    module = load_module(
        "engine_r2_storage", "src/production/engine/r2_storage.py"
    )
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{
        "Contents": [
            {"Key": "dataset/Zosterops lateralis/b.wav"},
            {"Key": "dataset/Gymnorhina tibicen/a.wav"},
            {"Key": "dataset/Gymnorhina tibicen/"},
        ]
    }]

    with patch.object(module.boto3, "client", return_value=client):
        storage = module.R2Storage()
        assert storage.list_species() == [
            "Gymnorhina tibicen",
            "Zosterops lateralis",
        ]


def test_simulator_groups_and_downloads_audio(r2_environment):
    module = load_module(
        "simulator_r2_storage", "src/production/simulator/src/r2_storage.py"
    )
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{
        "Contents": [
            {"Key": "dataset/Gymnorhina tibicen/a.wav"},
            {"Key": "dataset/Gymnorhina tibicen/b.wav"},
        ]
    }]
    client.get_object.return_value = {"Body": MagicMock(read=lambda: b"RIFF")}

    with patch.object(module.boto3, "client", return_value=client):
        storage = module.R2Storage()
        assert storage.list_audio_by_species() == {
            "Gymnorhina tibicen": [
                "dataset/Gymnorhina tibicen/a.wav",
                "dataset/Gymnorhina tibicen/b.wav",
            ]
        }
        assert storage.download_bytes("dataset/Gymnorhina tibicen/a.wav") == b"RIFF"


def test_missing_configuration_has_clear_error(monkeypatch):
    module = load_module(
        "engine_r2_storage_missing", "src/production/engine/r2_storage.py"
    )
    for name in (
        "R2_ACCOUNT_ID",
        "R2_BUCKET_NAME",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="Missing required R2 environment variables"):
        module.R2Storage()
