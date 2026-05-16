"""
EfficientNetV2 predictor utilities for Project Echo Engine integration.

This file loads a saved EfficientNetV2 model, class mapping, and preprocessing
configuration, then runs inference on a single audio file.

Expected saved files:
    _trained_models/
        efficientnetv2_project_echo.pt
        class_mapping.json
        preprocess_config.json
        training_metrics.json

Main functions:
    load_model()
    load_class_mapping()
    load_preprocess_config()
    preprocess_audio()
    predict_audio()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
import torch
import timm


def load_class_mapping(class_mapping_path: str | Path) -> Dict[str, Any]:
    """
    Load the class mapping JSON file.

    The expected format is:
        {
            "classes": [...],
            "label_to_index": {...},
            "index_to_label": {"0": "...", "1": "..."}
        }
    """
    class_mapping_path = Path(class_mapping_path)

    if not class_mapping_path.exists():
        raise FileNotFoundError(f"Class mapping file not found: {class_mapping_path}")

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_preprocess_config(preprocess_config_path: str | Path) -> Dict[str, Any]:
    """
    Load the preprocessing configuration JSON file.

    This should contain the same audio preprocessing settings used during training.
    """
    preprocess_config_path = Path(preprocess_config_path)

    if not preprocess_config_path.exists():
        raise FileNotFoundError(f"Preprocessing config file not found: {preprocess_config_path}")

    with open(preprocess_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(
    model_path: str | Path,
    device: Optional[str] = None
) -> Tuple[torch.nn.Module, str]:
    """
    Load the saved EfficientNetV2 model.

    Args:
        model_path: Path to efficientnetv2_project_echo.pt
        device: Optional device string, e.g. "cpu" or "cuda"

    Returns:
        model: Loaded PyTorch model in eval mode
        device: Device used for inference
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(model_path, map_location=device)

    required_keys = ["model_state_dict", "model_name", "num_classes", "in_chans"]
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        raise KeyError(f"Model checkpoint is missing required keys: {missing_keys}")

    model = timm.create_model(
        checkpoint["model_name"],
        pretrained=False,
        num_classes=checkpoint["num_classes"],
        in_chans=checkpoint["in_chans"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, device


def load_audio_fixed(
    audio_path: str | Path,
    target_sr: int,
    duration_s: float
) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono, resample it, and crop/pad to a fixed duration.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)

    target_len = int(duration_s * target_sr)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    return y, target_sr


def preprocess_audio(
    audio_path: str | Path,
    preprocess_config: Dict[str, Any]
) -> torch.Tensor:
    """
    Convert an audio file into the Mel spectrogram tensor expected by EfficientNetV2.

    Output shape:
        [1, 1, n_mels, time]
    """
    required_keys = ["target_sr", "duration_s", "n_mels", "hop_length", "fmin", "fmax"]
    missing_keys = [key for key in required_keys if key not in preprocess_config]
    if missing_keys:
        raise KeyError(f"Preprocessing config is missing required keys: {missing_keys}")

    y, sr = load_audio_fixed(
        audio_path,
        target_sr=preprocess_config["target_sr"],
        duration_s=preprocess_config["duration_s"],
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=preprocess_config["n_mels"],
        hop_length=preprocess_config["hop_length"],
        fmin=preprocess_config["fmin"],
        fmax=preprocess_config["fmax"],
    )

    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Same per-sample normalisation used in the training notebook.
    mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)

    # Shape: [batch, channel, n_mels, time]
    tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor


@torch.no_grad()
def predict_audio(
    audio_path: str | Path,
    model: torch.nn.Module,
    class_mapping: Dict[str, Any],
    preprocess_config: Dict[str, Any],
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Predict the species/class for one audio file.

    Returns:
        {
            "audio_path": "...",
            "predicted_index": int,
            "predicted_label": str,
            "confidence": float,
            "top_predictions": [
                {"index": int, "label": str, "confidence": float},
                ...
            ]
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if "index_to_label" not in class_mapping:
        raise KeyError("class_mapping must contain 'index_to_label'.")

    index_to_label = class_mapping["index_to_label"]

    x = preprocess_audio(audio_path, preprocess_config)
    x = x.to(device)

    outputs = model(x)
    probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()[0]

    pred_index = int(np.argmax(probs))
    confidence = float(probs[pred_index])
    pred_label = index_to_label[str(pred_index)]

    # Top 5 predictions, or fewer if there are fewer than 5 classes.
    top_k = min(5, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]

    top_predictions = [
        {
            "index": int(idx),
            "label": index_to_label[str(int(idx))],
            "confidence": float(probs[idx]),
        }
        for idx in top_indices
    ]

    return {
        "audio_path": str(audio_path),
        "predicted_index": pred_index,
        "predicted_label": pred_label,
        "confidence": confidence,
        "top_predictions": top_predictions,
    }


def load_predictor_bundle(
    model_path: str | Path,
    class_mapping_path: str | Path,
    preprocess_config_path: str | Path,
    device: Optional[str] = None,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any], str]:
    """
    Convenience function to load model, class mapping, and preprocessing config together.
    """
    model, device = load_model(model_path, device=device)
    class_mapping = load_class_mapping(class_mapping_path)
    preprocess_config = load_preprocess_config(preprocess_config_path)

    return model, class_mapping, preprocess_config, device
