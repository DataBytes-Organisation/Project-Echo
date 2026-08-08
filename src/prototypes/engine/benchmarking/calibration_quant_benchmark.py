from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Calibration metrics

def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> float:
    """Top-label (confidence) expected calibration error

    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|, over `n_bins` equal-width
    confidence bins. Lower = better, 0 = perfectly calibrated

    Args:
        probs:  (N, C) predicted class probabilities (rows sum to ~1)
        labels: (N,)   integer ground-truth class indices
        n_bins: number of equal-width bins over [0, 1]
    """

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    n = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi) if hi < 1.0 else (  # last bin is closed on the right so confidence==1.0 is counted
            (confidences > lo) & (confidences <= hi + 1e-9)
        )
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = confidences[in_bin].mean()
        bin_acc = accuracies[in_bin].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: Optional[int] = None) -> float:
    """Multi-class Brier score = mean squared error between the predicted
    probability vector and the one-hot label. Range [0, 2], lower = better"""

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    c = num_classes or probs.shape[1]
    onehot = np.zeros((len(labels), c), dtype=np.float64)
    onehot[np.arange(len(labels)), labels] = 1.0

    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_confidence, bin_accuracy, bin_weight) for a reliability diagram"""

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bconf, bacc, bw = [], [], []
    n = len(labels)

    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi + (1e-9 if hi >= 1.0 else 0.0))
        cnt = int(in_bin.sum())
        if cnt == 0:
            bconf.append(np.nan); bacc.append(np.nan); bw.append(0.0)
        else:
            bconf.append(conf[in_bin].mean())
            bacc.append(acc[in_bin].mean())
            bw.append(cnt / n)

    return np.array(bconf), np.array(bacc), np.array(bw)


def evaluate_predictions(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """Accuracy, macro-F1, ECE, and Brier"""
    from sklearn.metrics import f1_score

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
    }


# TFLite conversion + inference

TFLITE_MODES = ("fp32", "dynamic", "float16", "int8")


def convert_to_tflite(
    keras_model,
    mode: str = "fp32",
    representative_data: Optional[np.ndarray] = None,
    n_rep_samples: int = 100,
) -> bytes:
    """Convert a Keras model to a TFLite flatbuffer.

    mode:
        "fp32"     — no quantisation (baseline).
        "dynamic"  — dynamic-range int8 weight quantisation (activations float).
        "float16"  — float16 weight quantisation.
        "int8"     — full integer quantisation (needs `representative_data`);
                     int8 input & output tensors.
    """

    import tensorflow as tf

    if mode not in TFLITE_MODES:
        raise ValueError(f"mode must be one of {TFLITE_MODES}, got {mode!r}")

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if mode == "fp32":
        pass
    elif mode == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif mode == "int8":
        if representative_data is None:
            raise ValueError("int8 mode requires representative_data")
        rep = np.asarray(representative_data, dtype=np.float32)
        rep = rep[:n_rep_samples]

        def rep_gen():
            for i in range(len(rep)):
                yield [rep[i : i + 1]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter.convert()


def tflite_infer(tflite_model: bytes, X: np.ndarray) -> np.ndarray:
    """Run a TFLite model over X (N, ...) one sample at a time, handling
    quantised int8 input/output tensors, and return float probabilities."""

    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp = inp.get("quantization", (0.0, 0))
    out_scale, out_zp = out.get("quantization", (0.0, 0))
    X = np.asarray(X, dtype=np.float32)

    results = []

    for i in range(len(X)):
        x = X[i : i + 1]
        if inp["dtype"] in (np.int8, np.uint8):
            x = np.round(x / in_scale + in_zp).astype(inp["dtype"])
        else:
            x = x.astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if out["dtype"] in (np.int8, np.uint8):
            y = (y.astype(np.float32) - out_zp) * out_scale
        results.append(y.astype(np.float32))
    probs = np.array(results, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0

    return probs / row_sums


def tflite_latency_ms(tflite_model: bytes, X: np.ndarray, n: int = 50) -> float:
    """Mean single-sample inference latency in milliseconds (over up to n samples)."""
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    in_scale, in_zp = inp.get("quantization", (0.0, 0))
    X = np.asarray(X[:n], dtype=np.float32)

    # prepare samples once (exclude prep from timing)
    samples = []
    for i in range(len(X)):
        x = X[i : i + 1]
        if inp["dtype"] in (np.int8, np.uint8):
            x = np.round(x / in_scale + in_zp).astype(inp["dtype"])
        else:
            x = x.astype(inp["dtype"])
        samples.append(x)

    # warm-up
    interp.set_tensor(inp["index"], samples[0]); interp.invoke()

    t0 = time.perf_counter()
    for x in samples:
        interp.set_tensor(inp["index"], x)
        interp.invoke()
    dt = time.perf_counter() - t0
    return float(dt / len(samples) * 1000.0)


# Orchestration

@dataclass
class BenchmarkConfig:
    modes: Sequence[str] = field(default_factory=lambda: list(TFLITE_MODES))
    n_bins: int = 15
    n_rep_samples: int = 100
    latency_samples: int = 50
    out_dir: str = "results"


def benchmark_variants(
    keras_model,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
    config: Optional[BenchmarkConfig] = None,
):
    """Evaluate a Keras model and its TFLite variants on accuracy + calibration
    + size + latency. Writes a CSV + reliability plots to `config.out_dir` and
    returns a pandas DataFrame (one row per variant, plus a keras_fp32 ref)."""
    import pandas as pd

    cfg = config or BenchmarkConfig()
    os.makedirs(cfg.out_dir, exist_ok=True)
    X_eval = np.asarray(X_eval, dtype=np.float32)
    y_eval = np.asarray(y_eval).astype(int)

    rows = []
    probs_by_variant: Dict[str, np.ndarray] = {}

    # Reference: the original Keras model
    keras_probs = _as_prob(keras_model.predict(X_eval, verbose=0))
    ref = evaluate_predictions(keras_probs, y_eval, cfg.n_bins)
    ref.update(variant="keras_fp32", size_kb=_keras_size_kb(keras_model), latency_ms=np.nan)
    rows.append(ref)
    probs_by_variant["keras_fp32"] = keras_probs

    for mode in cfg.modes:
        try:
            tfl = convert_to_tflite(
                keras_model, mode,
                representative_data=X_eval if mode == "int8" else None,
                n_rep_samples=cfg.n_rep_samples,
            )
            probs = tflite_infer(tfl, X_eval)
            m = evaluate_predictions(probs, y_eval, cfg.n_bins)
            m.update(
                variant=f"tflite_{mode}",
                size_kb=round(len(tfl) / 1024.0, 1),
                latency_ms=round(tflite_latency_ms(tfl, X_eval, cfg.latency_samples), 3),
            )
            rows.append(m)
            probs_by_variant[f"tflite_{mode}"] = probs
        except Exception as e:  # keep going if one variant fails on this platform
            rows.append({"variant": f"tflite_{mode}", "error": str(e)[:200]})

    df = pd.DataFrame(rows)
    cols = ["variant", "accuracy", "macro_f1", "ece", "brier", "size_kb", "latency_ms"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
    csv_path = os.path.join(cfg.out_dir, "calibration_quant_results.csv")
    df.to_csv(csv_path, index=False)

    _plot_reliability_grid(probs_by_variant, y_eval, cfg.n_bins,
                           os.path.join(cfg.out_dir, "reliability_diagrams.png"))
    _plot_summary(df, os.path.join(cfg.out_dir, "summary_bars.png"))
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))
    return df


def _as_prob(y: np.ndarray) -> np.ndarray:
    """Ensure model output is a proper (N, C) probability matrix."""

    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] == 1:  # single logit -> treat as binary [1-p, p]
        y = np.concatenate([1.0 - y, y], axis=1)
    s = y.sum(axis=1, keepdims=True)

    # apply softmax if it doesn't already look normalised
    if not np.allclose(s, 1.0, atol=1e-2):
        e = np.exp(y - y.max(axis=1, keepdims=True))
        y = e / e.sum(axis=1, keepdims=True)

    return y


def _keras_size_kb(model) -> float:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as f:
        model.save(f.name)

        return round(os.path.getsize(f.name) / 1024.0, 1)


def _plot_reliability_grid(probs_by_variant, labels, n_bins, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(probs_by_variant.items())
    cols = min(3, len(items))
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    
    for ax in axes.flat:
        ax.axis("off")

    for i, (name, probs) in enumerate(items):
        ax = axes[i // cols][i % cols]; ax.axis("on")
        bconf, bacc, _ = reliability_curve(probs, labels, n_bins)
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
        ax.plot(bconf, bacc, "o-", color="#1f77b4", ms=4)
        ax.set_title(f"{name}\nECE={expected_calibration_error(probs, labels, n_bins):.4f}", fontsize=9)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _plot_summary(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[df.get("accuracy").notna()] if "accuracy" in df else df

    if d.empty:
        return

    metrics = [("accuracy", "Accuracy (higher=better)"), ("ece", "ECE (lower=better)"),
               ("brier", "Brier (lower=better)"), ("size_kb", "Size KB (lower=better)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, (col, title) in zip(axes, metrics):
        if col not in d:
            ax.axis("off"); continue
        ax.bar(d["variant"], d[col], color="#4c78a8")
        ax.set_title(title); ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# Generic Keras audio -> image preprocessing

def audio_to_image(path: str, sr: int = 16000, n_mels: int = 260, img_size: int = 260) -> np.ndarray:
    """Load an audio file and turn it into an (img_size, img_size, 3) float32
    mel-spectrogram 'image' in [0, 1]. Adjust params to match the target model."""

    import librosa
    from PIL import Image

    y, _sr = librosa.load(path, sr=sr, mono=True)

    if y.size == 0:
        y = np.zeros(sr, dtype=np.float32)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.ptp() + 1e-9)  # -> [0,1]
    img = Image.fromarray((mel_db * 255).astype(np.uint8)).resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0

    return np.stack([arr, arr, arr], axis=-1)


def build_eval_set(
    data_dir: str,
    class_names: Optional[Sequence[str]] = None,
    max_per_class: int = 20,
    exts: Tuple[str, ...] = (".wav", ".mp3", ".ogg"),
    **prep_kwargs,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build (X, y, class_names) from a folder-of-classes audio dataset.

    If `class_names` is given, y is indexed against THAT order (so it matches
    the target model's label order); folders not in the list are skipped.
    """
    import glob

    folders = sorted(d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d))
    names = list(class_names) if class_names else [os.path.basename(f) for f in folders]
    name_to_idx = {n: i for i, n in enumerate(names)}

    X, y = [], []
    for folder in folders:
        cname = os.path.basename(folder)

        if cname not in name_to_idx:
            continue
        files = [p for e in exts for p in glob.glob(os.path.join(folder, f"*{e}"))][:max_per_class]

        for p in files:
            try:
                X.append(audio_to_image(p, **prep_kwargs)); y.append(name_to_idx[cname])
            except Exception:
                continue

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), names


# EfficientNetV2 TFLite path (PR #948)

EFFICIENTNET_LABEL_ALIASES = {
    "Asio_flammeus": "sheowl",
    "Branta_bernicla_nigricans": "brant",
    "Horornis_diphone": "jabwar",
    "Meleagris_gallopavo": "wiltur",
    "Spilopelia_chinensis": "spodov",
}


def load_efficientnet_bundle(model_dir: str):
    """Load the EfficientNetV2 TFLite interpreter + class mapping + preprocess
    config from a directory containing the three artifacts shipped in PR #948."""
    import json
    import tensorflow as tf

    model_path = os.path.join(model_dir, "efficientnetv2_project_echo.tflite")
    with open(os.path.join(model_dir, "class_mapping.json")) as f:
        class_mapping = json.load(f)
    with open(os.path.join(model_dir, "preprocess_config.json")) as f:
        pre = json.load(f)

    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    index_to_label = class_mapping["index_to_label"]

    # class index -> name, ordered by integer index
    names = [index_to_label[str(i)] for i in range(len(index_to_label))]

    return interp, names, pre, os.path.getsize(model_path) / 1024.0


def efficientnet_preprocess(path: str, pre: dict) -> np.ndarray:
    """Audio file -> (1, n_mels, time) float32 tensor, matching production
    echo_engine_iot.efficientnetv2_preprocess_audio_bytes exactly."""
    import librosa

    target_sr = int(pre["target_sr"])
    duration_s = float(pre["duration_s"])
    n_mels = int(pre["n_mels"])
    hop_length = int(pre["hop_length"])
    fmin = float(pre["fmin"])
    fmax = float(pre["fmax"])

    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    audio = audio.astype(np.float32)

    target_length = int(target_sr * duration_s)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode="constant")
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_mels=n_mels,
        hop_length=hop_length, fmin=fmin, fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    # per-sample standardisation
    mel_db = (mel_db - float(np.mean(mel_db))) / (float(np.std(mel_db)) + 1e-6)
    return mel_db[np.newaxis, :, :]  # (1, n_mels, time) — channel dim added at infer


def efficientnet_predict_proba(interp, X: np.ndarray) -> np.ndarray:
    """Run the EfficientNetV2 TFLite interpreter over X (N, 1, n_mels, time),
    one sample at a time, applying a numerically-stable softmax to the logits
    (matching production). Returns (N, C) probabilities."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    expected = tuple(int(v) for v in inp["shape"])  # e.g. (1, 1, 128, 313)

    probs = []
    for i in range(len(X)):
        x = X[i][np.newaxis, ...].astype(inp["dtype"])  # (1, 1, n_mels, time)
        if tuple(x.shape) != expected:
            x = np.transpose(x, (0, 2, 3, 1))  # NHWC fallback, mirrors production
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        logits = interp.get_tensor(out["index"])[0].astype(np.float64)
        z = logits - logits.max()
        e = np.exp(z)
        probs.append(e / e.sum())
    return np.asarray(probs, dtype=np.float64)


def evaluate_efficientnet(
    model_dir: str,
    data_dir: str,
    max_per_class: int = 20,
    n_bins: int = 15,
    out_dir: str = "results_efficientnet",
    exts: Tuple[str, ...] = (".wav", ".mp3", ".ogg"),
):
    """End-to-end calibration eval of the production EfficientNetV2 TFLite model
    on a folder-per-class audio dataset. Writes CSV + reliability diagram and
    returns (metrics_dict, probs, labels)."""
    import glob
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    interp, names, pre, size_kb = load_efficientnet_bundle(model_dir)
    name_to_idx = {n: i for i, n in enumerate(names)}

    X, y, used = [], [], set()
    folders = sorted(d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d))
    for folder in folders:
        cname = os.path.basename(folder)
        model_name = EFFICIENTNET_LABEL_ALIASES.get(cname, cname)
        if model_name not in name_to_idx:
            continue  # folder isn't one of the model's classes (e.g. 'spectrograms')
        idx = name_to_idx[model_name]
        files = [p for e in exts for p in glob.glob(os.path.join(folder, f"*{e}"))][:max_per_class]
        for p in files:
            try:
                X.append(efficientnet_preprocess(p, pre))
                y.append(idx)
                used.add(idx)
            except Exception:
                continue

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    print(f"Eval set: {len(X)} clips across {len(used)} of {len(names)} model classes")
    if len(X) == 0:
        raise SystemExit("No evaluable clips — check dataset path / labels.")

    t0 = time.perf_counter()
    probs = efficientnet_predict_proba(interp, X)
    latency_ms = (time.perf_counter() - t0) / len(X) * 1000.0

    metrics = evaluate_predictions(probs, y, n_bins)
    metrics.update(variant="efficientnetv2_tflite_fp32",
                   size_kb=round(size_kb, 1),
                   latency_ms=round(latency_ms, 3),
                   n_samples=len(X), n_classes_seen=len(used))

    df = pd.DataFrame([metrics])
    cols = ["variant", "accuracy", "macro_f1", "ece", "brier",
            "size_kb", "latency_ms", "n_samples", "n_classes_seen"]
    df = df[[c for c in cols if c in df.columns]]
    csv_path = os.path.join(out_dir, "efficientnet_calibration_results.csv")
    df.to_csv(csv_path, index=False)
    _plot_reliability_grid({"efficientnetv2_tflite_fp32": probs}, y, n_bins,
                           os.path.join(out_dir, "efficientnet_reliability.png"))
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))
    return metrics, probs, y


# CLI

def _load_keras(model_path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(model_path)


def _selftest():
    """Prove the whole pipeline end-to-end on a tiny synthetic model + data."""
    import tensorflow as tf

    print("Self-test: building tiny model + synthetic data ...")
    n, c, hw = 200, 5, 32
    rng = np.random.default_rng(0)
    X = rng.random((n, hw, hw, 3)).astype(np.float32)
    y = rng.integers(0, c, size=n)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((hw, hw, 3)),
        tf.keras.layers.Conv2D(8, 3, activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(c, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(X, y, epochs=1, verbose=0)

    df = benchmark_variants(model, X, y,
                            config=BenchmarkConfig(out_dir="results_selftest", latency_samples=20))
    assert "ece" in df.columns and df["ece"].notna().any(), "ECE not computed"
    print("\nSelf-test OK ✓  (framework, metrics, TFLite conversion & eval all working)")


def main():
    ap = argparse.ArgumentParser(description="Calibration & TFLite quantisation benchmark")
    ap.add_argument("--selftest", action="store_true", help="run synthetic end-to-end check")
    ap.add_argument("--efficientnet", metavar="MODEL_DIR",
                    help="dir with efficientnetv2_project_echo.tflite + class_mapping.json "
                         "+ preprocess_config.json (production model, PR #948)")
    ap.add_argument("--model", help="path to Keras .h5 / SavedModel dir")
    ap.add_argument("--data", help="dataset dir (folder per class of audio)")
    ap.add_argument("--labels", help="optional text file: one class name per line, in model order")
    ap.add_argument("--img-size", type=int, default=260)
    ap.add_argument("--n-mels", type=int, default=260)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--max-per-class", type=int, default=20)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--modes", nargs="+", default=list(TFLITE_MODES))
    args = ap.parse_args()

    if args.selftest:
        _selftest(); return
    if args.efficientnet:
        if not args.data:
            ap.error("--efficientnet requires --data")
        evaluate_efficientnet(args.efficientnet, args.data,
                              max_per_class=args.max_per_class,
                              n_bins=15, out_dir=args.out_dir)
        return
    if not (args.model and args.data):
        ap.error("provide --efficientnet+--data, --model+--data, or --selftest")

    class_names = None
    if args.labels:
        with open(args.labels) as f:
            class_names = [ln.strip() for ln in f if ln.strip()]

    print("Loading model:", args.model)
    model = _load_keras(args.model)
    print("Building eval set from:", args.data)
    X, y, names = build_eval_set(args.data, class_names, args.max_per_class,
                                 sr=args.sr, n_mels=args.n_mels, img_size=args.img_size)
    print(f"Eval set: {len(X)} samples across {len(set(y.tolist()))} classes")
    benchmark_variants(model, X, y, names,
                       config=BenchmarkConfig(modes=args.modes, out_dir=args.out_dir))


if __name__ == "__main__":
    main()
