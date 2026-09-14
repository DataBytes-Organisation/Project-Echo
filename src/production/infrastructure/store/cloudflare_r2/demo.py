"""Run the prototype engine and simulator behaviours against a real R2 bucket."""

from __future__ import annotations

import hashlib

from .prototype_integration import R2EnginePrototype, R2SimulatorPrototype
from .r2_storage import R2Config, R2Storage


def main() -> None:
    config = R2Config.from_env()
    storage = R2Storage(config)
    storage.check_connection()
    print("R2 connection: PASS")

    engine = R2EnginePrototype(storage, config.dataset_prefix)
    engine_species = engine.load_species_list()
    print(f"Engine species loading: PASS ({len(engine_species)} species)")
    print("Engine species:", engine_species)

    simulator = R2SimulatorPrototype(storage, config.dataset_prefix)
    simulator_species = simulator.load_species_list()
    if simulator_species != engine_species:
        raise RuntimeError("Engine and simulator species lists do not match")
    print("Engine/simulator species agreement: PASS")

    selected_species = simulator_species[0]
    sample = simulator.download_random_audio(selected_species)
    digest = hashlib.sha256(sample.audio_bytes).hexdigest()
    print(f"Simulator random audio download: PASS ({sample.file_name})")
    print(f"Downloaded bytes: {len(sample.audio_bytes)}")
    print(f"SHA-256: {digest}")
    print("Prototype integration: PASS")


if __name__ == "__main__":
    main()
