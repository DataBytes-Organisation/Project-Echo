"""Populate the Backend's real `detections` collection with real detections
that include a real model embedding, by running actual inference on real
local audio files and POSTing each result to the Backend's own HTTP API
(POST /detections) - the same integration boundary the real production
engine (src/production/engine/echo_engine.py) uses, rather than importing
torch into the Backend process directly (Backend has no ML dependencies
installed; Engine and Backend are separate services that talk over HTTP by
design in this codebase).

Prerequisites:
    1. MongoDB + Redis running:
       docker compose -f src/deployment/docker/docker-compose.yml up echo_store echo-redis -d
    2. The backend running locally (see docs/team-guides/TDD_Guide.md section 4
       for the exact env vars/command).

Run:
    .venv\\Scripts\\python.exe populate_detections_with_embeddings.py
"""

import base64
import random
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path.cwd()))

import requests
import torch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

BACKEND_URL = "http://127.0.0.1:9000"
CHECKPOINT_DIR = Path.cwd() / "outputs" / "sprint_hd_pretrained_long_run"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_efficientnet_v2.pth"
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.txt"
DATA_DIR = Path.cwd().parents[1] / "data_files"

N_SPECIES = 8
FILES_PER_SPECIES = 3


def load_model():
    CONFIG_DIR = str(Path.cwd() / "config")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["model=efficientnet_v2", "model.norm_choice=keep_bn"])

    class_names = CLASS_NAMES_PATH.read_text().splitlines()
    OmegaConf.set_struct(cfg, False)
    cfg.data.num_classes = len(class_names)
    OmegaConf.set_struct(cfg, True)

    from model import Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(cfg)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, class_names, cfg, device


def build_preprocessor(cfg, device):
    import torchaudio
    import torch.nn.functional as F
    import math

    target_sample_rate = cfg.data.sample_rate
    target_samples = int(cfg.data.audio_clip_duration * target_sample_rate)
    common_source_sr = 44100
    resampler = torchaudio.transforms.Resample(orig_freq=common_source_sr, new_freq=target_sample_rate)
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.data.sample_rate,
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        n_mels=cfg.data.n_mels,
        f_min=cfg.data.fmin,
        f_max=cfg.data.fmax,
        power=2.0,
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=cfg.data.top_db)
    top_db = cfg.data.top_db

    def preprocess(file_path):
        waveform, sr = torchaudio.load(str(file_path))
        if sr != target_sample_rate:
            waveform = resampler(waveform) if sr == common_source_sr else torchaudio.functional.resample(waveform, sr, target_sample_rate)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        num_samples = waveform.shape[1]
        if num_samples < target_samples:
            repeats = math.ceil(target_samples / num_samples)
            waveform = waveform.repeat(1, repeats)
            num_samples = waveform.shape[1]
        start = (num_samples - target_samples) // 2
        waveform = waveform[:, start : start + target_samples]

        spec = mel_spec(waveform)
        spec = amplitude_to_db(spec)
        spec = (spec + top_db) / top_db
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        return spec.unsqueeze(0).to(device)

    return preprocess


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found at {CHECKPOINT_PATH} - has the keep_bn retraining run finished?")
        return

    model, class_names, cfg, device = load_model()
    preprocess = build_preprocessor(cfg, device)

    species_dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    species_by_count = sorted(species_dirs, key=lambda d: sum(1 for _ in d.glob("*")), reverse=True)
    chosen_species = species_by_count[:N_SPECIES]

    created_ids = []
    random.seed(0)

    for species_dir in chosen_species:
        files = list(species_dir.glob("*"))
        random.shuffle(files)
        for f in files[:FILES_PER_SPECIES]:
            try:
                spec = preprocess(f)
                with torch.no_grad():
                    logits = model(spec)
                    probs = torch.softmax(logits, dim=1)
                    confidence, predicted_idx = probs.max(dim=1)
                    embedding = model.model.get_embedding(spec).squeeze(0).cpu().tolist()

                predicted_species = class_names[predicted_idx.item()]
                audio_bytes = f.read_bytes()
                audio_b64 = base64.b64encode(audio_bytes[:200000]).decode("ascii")  # cap payload size

                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "sensorId": "hd-feature-demo",
                    "species": predicted_species,
                    "microphoneLLA": [-33.11, 150.06, 23],
                    "animalEstLLA": [-33.11, 150.06, 23],
                    "animalTrueLLA": [-33.11, 150.06, 23],
                    "animalLLAUncertainty": 10,
                    "audioClip": audio_b64,
                    "confidence": max(0.01, min(99.99, float(confidence.item()) * 100)),
                    "sampleRate": cfg.data.sample_rate,
                    "embedding": embedding,
                }

                response = requests.post(f"{BACKEND_URL}/detections", json=payload, timeout=30)
                if response.status_code == 200:
                    detection_id = response.json().get("_id") or response.json().get("id")
                    created_ids.append(detection_id)
                    print(f"Created detection {detection_id}: file={f.name} true_species={species_dir.name} predicted={predicted_species}")
                else:
                    print(f"Failed to create detection for {f.name}: {response.status_code} {response.text[:200]}")
            except Exception as e:
                print(f"Skipped {f.name}: {e}")

    print(f"\nCreated {len(created_ids)} real detections with embeddings.")
    if created_ids:
        print(f"Try: GET {BACKEND_URL}/detections/{created_ids[0]}/similar")


if __name__ == "__main__":
    main()
