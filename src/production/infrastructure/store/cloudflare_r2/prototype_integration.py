"""Prototype-only R2 integrations matching Project Echo production behaviours."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol


class DatasetStorage(Protocol):
    def list_species(self, prefix: str | None = None) -> list[str]: ...

    def group_objects_by_species(
        self, prefix: str | None = None
    ) -> dict[str, list[str]]: ...

    def download_bytes(self, key: str) -> bytes: ...


@dataclass(frozen=True)
class SimulatorAudioSample:
    species_name: str
    object_key: str
    file_name: str
    audio_bytes: bytes


class R2EnginePrototype:
    """R2 equivalent of EchoEngine.gcp_load_species_list()."""

    def __init__(self, storage: DatasetStorage, prefix: str | None = None) -> None:
        self.storage = storage
        self.prefix = prefix

    def load_species_list(self) -> list[str]:
        species = self.storage.list_species(self.prefix)
        if not species:
            raise RuntimeError("No species audio was found in the configured R2 prefix")
        return species


class R2SimulatorPrototype:
    """R2 equivalent of simulator species loading and random audio download."""

    def __init__(
        self,
        storage: DatasetStorage,
        prefix: str | None = None,
        random_source: random.Random | None = None,
    ) -> None:
        self.storage = storage
        self.prefix = prefix
        self.random_source = random_source or random.Random()
        self.audio_objects: dict[str, list[str]] = {}

    def load_species_list(self) -> list[str]:
        self.audio_objects = self.storage.group_objects_by_species(self.prefix)
        if not self.audio_objects:
            raise RuntimeError("No species audio was found in the configured R2 prefix")
        return list(self.audio_objects)

    def download_random_audio(self, species_name: str) -> SimulatorAudioSample:
        if not self.audio_objects:
            self.load_species_list()
        keys = self.audio_objects.get(species_name)
        if not keys:
            raise KeyError(f"No R2 audio objects found for species: {species_name}")

        key = self.random_source.choice(keys)
        audio = self.storage.download_bytes(key)
        if not audio:
            raise RuntimeError(f"R2 object is empty: {key}")

        return SimulatorAudioSample(
            species_name=species_name,
            object_key=key,
            file_name=PurePosixPath(key).name,
            audio_bytes=audio,
        )
