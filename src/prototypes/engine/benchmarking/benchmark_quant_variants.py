"""
Benchmark the EfficientNetV2 TFLite quantisation sweep.

Scores every `*.tflite` in a directory on a folder-per-class audio dataset and
reports accuracy, macro-F1, ECE, Brier, size and per-clip latency — the
fp32-vs-float16-vs-int8 comparison for Sprint 2.

Preprocessing mirrors production `efficientnetv2_preprocess_audio_bytes`
exactly (and `calibration_quant_benchmark.efficientnet_preprocess`). Full-integer
(int8 I/O) variants are handled by quantising the input and dequantising the
output from the tflite quantisation params, so they can be scored alongside the
float-I/O variants.

Run this in an env with a modern TensorFlow (the same env used to build the
variants). Preprocessing needs librosa. To avoid re-decoding audio per variant,
pass --cache-npy to preprocess once and reuse.

Usage:
    python benchmark_quant_variants.py \
        --variants-dir results_efficientnet/variants \
        --data /path/to/audio_dataset \
        --model-dir ../../production/engine/models/efficientnetv2 \
        --max-per-class 15 \
        --out-csv results_efficientnet/quant_sweep_results.csv
"""

import argparse
import csv
import glob
import json
import os
import time

import numpy as np


# folders whose names differ from the model's class labels (shared with the
# main calibration harness)
LABEL_ALIASES = {
    "Asio_flammeus": "sheowl",
    "Branta_bernicla_nigricans": "brant",
    "Horornis_diphone": "jabwar",
    "Meleagris_gallopavo": "wiltur",
    "Spilopelia_chinensis": "spodov",
}


def load_bundle(model_dir):
    """Read class names + preprocess config from the production model bundle."""
    with open(os.path.join(model_dir, "class_mapping.json")) as f:
        index_to_label = json.load(f)["index_to_label"]
    with open(os.path.join(model_dir, "preprocess_config.json")) as f:
        pre = json.load(f)
    names = [index_to_label[str(i)] for i in range(len(index_to_label))]
    return names, pre


def preprocess(path, pre):
    """Audio file -> (1, n_mels, time) float32, matching production exactly."""
    import librosa

    sr = int(pre["target_sr"])
    dur = float(pre["duration_s"])
    n_mels = int(pre["n_mels"])
    hop = int(pre["hop_length"])
    fmin = float(pre["fmin"])
    fmax = float(pre["fmax"])

    audio, _ = librosa.load(path, sr=sr, mono=True)
    audio = audio.astype(np.float32)
    tl = int(sr * dur)
    audio = np.pad(audio, (0, tl - len(audio))) if len(audio) < tl else audio[:tl]

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                         hop_length=hop, fmin=fmin, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_db = (mel_db - float(mel_db.mean())) / (float(mel_db.std()) + 1e-6)
    return mel_db[np.newaxis, :, :]


def build_eval_set(data_dir, names, pre, max_per_class):
    name_to_idx = {n: i for i, n in enumerate(names)}
    X, y, used = [], [], set()
    for folder in sorted(d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)):
        cname = os.path.basename(folder)
        mname = LABEL_ALIASES.get(cname, cname)
        if mname not in name_to_idx:
            continue
        idx = name_to_idx[mname]
        files = [p for e in (".wav", ".mp3", ".ogg")
                 for p in glob.glob(os.path.join(folder, f"*{e}"))][:max_per_class]
        for p in files:
            try:
                X.append(preprocess(p, pre)); y.append(idx); used.add(idx)
            except Exception:
                continue
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), used


def build_eval_set_from_manifest(data_dir, names, pre, manifest_csv):
    """Score exactly the files in a shared held-out manifest.

    The manifest has the columns `filepath,species,label_id,split` (see the
    engine evaluation `heldout_manifest.csv`); `filepath` is POSIX-relative to
    `data_dir`. Missing files are skipped and reported, so a partially-present
    dataset still yields a clean held-out score.
    """
    name_to_idx = {n: i for i, n in enumerate(names)}
    X, y, used, missing = [], [], set(), 0
    with open(manifest_csv, newline="") as f:
        for row in csv.DictReader(f):
            mname = LABEL_ALIASES.get(row["species"], row["species"])
            if mname not in name_to_idx:
                continue
            path = os.path.join(data_dir, row["filepath"])
            if not os.path.exists(path):
                missing += 1
                continue
            idx = int(row["label_id"])
            try:
                X.append(preprocess(path, pre)); y.append(idx); used.add(idx)
            except Exception:
                continue
    if missing:
        print(f"  (manifest: {missing} listed files not present locally, skipped)")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int), used


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def predict(interp, X):
    """Run one tflite over X (N,1,n_mels,time NCHW). Handles NHWC layout and
    int8 I/O. Returns (N, C) probabilities."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    int8 = inp["dtype"] == np.int8
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    nchw_in = tuple(inp["shape"])[1] == 1  # (1,1,mel,time) vs (1,mel,time,1)

    probs = []
    for i in range(len(X)):
        x = X[i][np.newaxis, ...]                 # (1,1,mel,time)
        if not nchw_in:
            x = np.transpose(x, (0, 2, 3, 1))     # -> NHWC
        if int8:
            x = np.round(x / s_in + z_in).clip(-128, 127).astype(np.int8)
        else:
            x = x.astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        o = interp.get_tensor(out["index"])[0].astype(np.float64)
        if int8:
            o = (o - z_out) * s_out
        probs.append(_softmax(o))
    return np.asarray(probs)


def metrics(probs, y, n_classes, n_bins=15):
    pred = probs.argmax(-1)
    conf = probs.max(-1)
    n = len(y)
    acc = float((pred == y).mean())

    f1s = []
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    ece = 0.0
    correct = (pred == y).astype(float)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / n

    oh = np.zeros((n, n_classes))
    oh[np.arange(n), y] = 1
    brier = float(((probs - oh) ** 2).sum(1).mean())
    return dict(accuracy=round(acc, 4), macro_f1=round(macro_f1, 4),
                ece=round(ece, 4), brier=round(brier, 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-dir", required=True)
    ap.add_argument("--model-dir", required=True,
                    help="production bundle dir (class_mapping.json + preprocess_config.json)")
    ap.add_argument("--data", help="folder-per-class audio dataset (or manifest root when --manifest is set)")
    ap.add_argument("--manifest", help="shared held-out manifest CSV (filepath,species,label_id,split); "
                                       "scores exactly its files relative to --data")
    ap.add_argument("--cache-npy", help="preprocess once to this prefix (writes _X.npy/_y.npy) and reuse")
    ap.add_argument("--max-per-class", type=int, default=15)
    ap.add_argument("--out-csv", default="results_efficientnet/quant_sweep_results.csv")
    args = ap.parse_args()

    import tensorflow as tf

    names, pre = load_bundle(args.model_dir)
    n_classes = len(names)

    xp = args.cache_npy + "_X.npy" if args.cache_npy else None
    yp = args.cache_npy + "_y.npy" if args.cache_npy else None
    if xp and os.path.exists(xp) and os.path.exists(yp):
        X, y = np.load(xp), np.load(yp)
        print(f"Loaded cached eval set: X={X.shape}")
    else:
        if not args.data:
            ap.error("--data is required unless a cached --cache-npy set exists")
        if args.manifest:
            X, y, used = build_eval_set_from_manifest(args.data, names, pre, args.manifest)
            print(f"Held-out eval set: {len(X)} clips across {len(used)}/{n_classes} classes "
                  f"(from {os.path.basename(args.manifest)})")
        else:
            X, y, used = build_eval_set(args.data, names, pre, args.max_per_class)
            print(f"Eval set: {len(X)} clips across {len(used)}/{n_classes} classes")
        if xp:
            np.save(xp, X); np.save(yp, y)

    rows = []
    for path in sorted(glob.glob(os.path.join(args.variants_dir, "*.tflite"))):
        vname = (os.path.basename(path)
                 .replace("efficientnetv2_project_echo_", "").replace(".tflite", ""))
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        t0 = time.perf_counter()
        probs = predict(interp, X)
        latency_ms = (time.perf_counter() - t0) / len(X) * 1000.0
        m = metrics(probs, y, n_classes)
        m.update(variant=vname, size_mb=round(os.path.getsize(path) / 1048576, 1),
                 latency_ms=round(latency_ms, 2))
        rows.append(m)
        print(m, flush=True)

    import csv
    order = ["variant", "accuracy", "macro_f1", "ece", "brier", "size_mb", "latency_ms"]
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=order)
        w.writeheader()
        w.writerows([{k: r[k] for k in order} for r in rows])
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    main()
