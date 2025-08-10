#!/usr/bin/env python3
"""
Minimal experiment: Baseline vs Real-Noise+RIR on ID and OOD.

Outputs:
  results/{arm}/seed_{s}/metrics.json, training.csv
  results/summary.csv
  results/significance.txt (only if ≥2 seeds and both arms)
"""

import os
import json
import time
import glob
import math
import random
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

warnings.filterwarnings("ignore")

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import fftconvolve
from scipy.stats import ttest_rel

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score

# Optional baseline ops (if installed)
try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    HAVE_AUDIOMENT = True
except Exception:
    HAVE_AUDIOMENT = False

np.set_printoptions(suppress=True, linewidth=120)

# ----------------------------- Config -----------------------------
@dataclass
class MelConfig:
    sr: int = 48000
    clip_seconds: float = 5.0
    n_fft: int = 2048
    hop_length: int = 200
    n_mels: int = 260
    fmin: int = 20
    fmax: int = 13000
    top_db: int = 80
    img_w: int = 260
    img_h: int = 260
    channels: int = 3

@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 20
    patience: int = 4
    steps_per_epoch: Optional[int] = None
    lr: float = 1e-4

@dataclass
class ArmConfig:
    name: str
    use_baseline: bool = False
    use_real_noise: bool = False
    use_rir: bool = False
    p_noise: float = 0.6
    p_rir: float = 0.6
    snr_choices: Tuple[int, ...] = (-5, 0, 5, 10, 15, 20)

DEFAULT_MEL = MelConfig()
DEFAULT_TRAIN = TrainConfig()

ARMS: Dict[str, ArmConfig] = {
    "baseline": ArmConfig("baseline", use_baseline=True),
    "noise_rir": ArmConfig("noise_rir", use_real_noise=True, use_rir=True),
}

EPS = 1e-9

# -------------------------- Utilities ----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass

def _is_readable_audio(path: str) -> bool:
    # Cheap header probe to filter corrupt/unreadable files
    try:
        sf.info(path)
        return True
    except Exception:
        return False

def list_audio_files(root: str) -> Tuple[List[str], List[str]]:
    """
    Restrict to WAV/FLAC to avoid audioread/ffmpeg on Windows.
    Filters out unreadable/corrupt files up-front.
    """
    if not os.path.isdir(root):
        raise ValueError(f"Directory not found: {root}")

    exts = (".wav", ".flac")
    files, labels = [], []
    try:
        subdirs = [d for d in sorted(next(os.walk(root))[1])]
    except StopIteration:
        subdirs = []

    if not subdirs:
        raise ValueError(f"No class subfolders found in {root}")

    for cls in subdirs:
        cdir = os.path.join(root, cls)
        for f in glob.glob(os.path.join(cdir, "**", "*"), recursive=True):
            if f.lower().endswith(exts):
                try:
                    sf.info(f)  # cheap header probe
                except Exception:
                    continue
                files.append(f)
                labels.append(cls)

    if len(files) == 0:
        raise ValueError(f"No readable WAV/FLAC audio found in {root}")
    return files, labels


def stratified_splits(X, y, seed: int, test_size=0.2, val_size=0.2):
    """Stratified test split; then stratified val if feasible; else random val (tiny-set safe)."""
    X = np.array(X)
    y = np.array(y)
    n = len(y)
    if n < 3:
        raise ValueError(f"Dataset too small for splits: {n} items")

    test_size_abs = max(1, int(round(test_size * n)))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size_abs, random_state=seed)
    (train_idx, test_idx) = next(sss1.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    from collections import Counter
    can_strat = all(v >= 2 for v in Counter(y_train).values())
    val_size_abs = max(1, int(round(val_size * len(y_train))))
    if len(y_train) - val_size_abs < 1:
        val_size_abs = max(1, len(y_train) - 1)

    if can_strat:
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_abs, random_state=seed)
        (train_idx2, val_idx) = next(sss2.split(X_train, y_train))
        X_tr, X_val = X_train[train_idx2], X_train[val_idx]
        y_tr, y_val = y_train[train_idx2], y_train[val_idx]
    else:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(X_train))
        val_idx = perm[:val_size_abs]
        train_idx2 = perm[val_size_abs:]
        if len(train_idx2) == 0:
            train_idx2 = perm[:1]
            val_idx = perm[1:]
        X_tr, X_val = X_train[train_idx2], X_train[val_idx]
        y_tr, y_val = y_train[train_idx2], y_train[val_idx]

    return list(X_tr), list(y_tr), list(X_val), list(y_val), list(X_test), list(y_test)


def _read_pcm_wav(path: str) -> Tuple[np.ndarray, int]:
    """Tiny PCM WAV reader via wave module; fallback if soundfile fails."""
    import wave, contextlib
    with contextlib.closing(wave.open(path, "rb")) as w:
        n_ch = w.getnchannels()
        sr = w.getframerate()
        n_frames = w.getnframes()
        data = w.readframes(n_frames)
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr

def load_audio(path: str, cfg: MelConfig) -> np.ndarray:
    """
    Decode via soundfile -> wave (PCM). Never touch audioread/ffmpeg.
    Only accept WAV/FLAC. Fail fast with path info if unreadable.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in {".wav", ".flac"}:
        raise ValueError(f"Unsupported format {ext} for {path}. Convert to WAV/FLAC.")

    sr_in = None
    try:
        y, sr_in = sf.read(path, dtype="float32", always_2d=False)
        if y is None:
            raise RuntimeError(f"sf.read returned None for {path}")
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        try:
            y, sr_in = _read_pcm_wav(path)
        except Exception as e:
            raise RuntimeError(f"Cannot decode {path} via soundfile or PCM WAV reader: {e}")

    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    if sr_in != cfg.sr:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=cfg.sr)

    tgt_len = int(cfg.clip_seconds * cfg.sr)
    if len(y) < tgt_len:
        y = np.pad(y, (0, tgt_len - len(y)))
    else:
        y = y[:tgt_len]

    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.astype(np.float32)


def mel_image(y: np.ndarray, cfg: MelConfig) -> np.ndarray:
    if y.ndim != 1:
        y = y.reshape(-1)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    m = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    m = librosa.power_to_db(m, top_db=cfg.top_db, ref=1.0)

    m = np.moveaxis(m, 0, 1)  # time x mel
    max_T = int(cfg.clip_seconds * cfg.sr / cfg.hop_length)
    m = m[:max_T, :]

    EPS = 1e-9
    m_min = np.nanmin(m)
    m = m - m_min
    m_max = np.nanmax(m)
    m = m / (m_max + EPS)

    m = np.expand_dims(m, -1)
    m = tf.image.resize(m, (cfg.img_h, cfg.img_w), method="lanczos5").numpy()
    m = np.repeat(m, cfg.channels, axis=-1)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    return m.astype(np.float32)


def specaugment(mel_img: np.ndarray, time_w=10, freq_w=6, n_time=1, n_freq=1) -> np.ndarray:
    x = mel_img.copy()
    T, F = x.shape[0], x.shape[1]
    if T <= 1 or F <= 1:
        return x
    for _ in range(max(0, int(n_time))):
        if T <= 1:
            break
        t = np.random.randint(0, min(time_w, T - 1) + 1)
        if t > 0:
            t0 = np.random.randint(0, max(1, T - t))
            x[t0:t0 + t, :, :] = 0.0
    for _ in range(max(0, int(n_freq))):
        if F <= 1:
            break
        f = np.random.randint(0, min(freq_w, F - 1) + 1)
        if f > 0:
            f0 = np.random.randint(0, max(1, F - f))
            x[:, f0:f0 + f, :] = 0.0
    return x

def _make_shift(cfg):
    """
    Build an audiomentations.Shift that works across versions:
    - Newer: Shift(min_fraction=..., max_fraction=...)
    - Some:  Shift(min_shift=..., max_shift=..., shift_unit="fraction")
    - Older: Shift(min_shift=..., max_shift=..., shift_unit="seconds")
    If all fail, return None (skip shift).
    """
    try:
        return Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2)
    except (TypeError, ValueError):
        pass
    try:
        return Shift(min_shift=-0.5, max_shift=0.5, shift_unit="fraction", p=0.2)
    except (TypeError, ValueError):
        pass
    try:
        max_shift_sec = 0.5 * float(getattr(cfg, "clip_seconds", 5.0))
        return Shift(min_shift=-max_shift_sec, max_shift=max_shift_sec, shift_unit="seconds", p=0.2)
    except Exception:
        return None

def apply_baseline_ops(y: np.ndarray, cfg: MelConfig) -> np.ndarray:
    if not HAVE_AUDIOMENT:
        return y

    shift_aug = _make_shift(cfg)

    augs = []
    try:
        augs.append(AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2))
    except Exception:
        pass
    try:
        augs.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2))
    except Exception:
        pass
    try:
        augs.append(PitchShift(min_semitones=-4, max_semitones=4, p=0.2))
    except Exception:
        pass
    if shift_aug is not None:
        augs.append(shift_aug)

    if not augs:
        return y

    out = Compose(augs)(samples=y, sample_rate=getattr(cfg, "sr", 48000)).astype(np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# Directory cache
_DIR_CACHE: Dict[Tuple[str, Tuple[str, ...]], List[str]] = {}
def pick_random_file(root: Optional[str], exts=(".wav", ".flac")) -> Optional[str]:
    if not root or not os.path.isdir(root):
        return None
    key = (root, exts)
    if key not in _DIR_CACHE:
        _DIR_CACHE[key] = [
            p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
            if p.lower().endswith(exts)
            and (lambda _p: (sf.info(_p) or True))(p)  # probe; throws if unreadable
        ]
    cand = _DIR_CACHE[key]
    return random.choice(cand) if cand else None

def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    L = len(signal)
    noise = noise[:L]
    if len(noise) < L:
        noise = np.pad(noise, (0, L - len(noise)))
    s_rms = float(np.sqrt(np.mean(signal ** 2) + EPS))
    n_rms = float(np.sqrt(np.mean(noise ** 2) + EPS))
    if n_rms < 1e-8:
        return signal.astype(np.float32)
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    noise_scaled = noise * (target_n_rms / (n_rms + EPS))
    out = signal + noise_scaled
    mx = float(np.max(np.abs(out)) + EPS)
    if mx > 1.0:
        out = out / mx
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

def pick_noise(noise_root: Optional[str], cfg: MelConfig) -> Optional[np.ndarray]:
    nf = pick_random_file(noise_root, exts=(".wav", ".flac"))
    if not nf:
        return None
    try:
        n, sr = sf.read(nf, dtype="float32", always_2d=False)
        if isinstance(n, np.ndarray) and n.ndim > 1:
            n = n.mean(axis=1)
    except Exception:
        try:
            n, sr = _read_pcm_wav(nf)
        except Exception as e:
            # Skip unreadable noise file
            return None
    if sr != cfg.sr:
        n = librosa.resample(n, orig_sr=sr, target_sr=cfg.sr)
    if len(n) < int(cfg.clip_seconds * cfg.sr):
        n = np.pad(n, (0, int(cfg.clip_seconds * cfg.sr) - len(n)))
    n = n[:int(cfg.clip_seconds * cfg.sr)]
    n = np.nan_to_num(n, nan=0.0, posinf=0.0, neginf=0.0)
    return n.astype(np.float32)

def convolve_rir(y: np.ndarray, rir: np.ndarray, wet: float) -> np.ndarray:
    z = fftconvolve(y, rir, mode="full")[:len(y)]
    out = wet * z + (1.0 - wet) * y
    mx = float(np.max(np.abs(out)) + EPS)
    if mx > 1.0:
        out = out / mx
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)

def parse_t60_bucket_from_name(fname: str) -> str:
    s = os.path.basename(fname).lower()
    import re
    m = re.search(r"t60[_=](\d+\.?\d*)s", s)
    if not m:
        return "unknown"
    t60 = float(m.group(1))
    if 0.2 <= t60 < 0.4:
        return "0.2-0.4"
    if 0.4 <= t60 < 0.8:
        return "0.4-0.8"
    if 0.8 <= t60 < 1.2:
        return "0.8-1.2"
    return "other"

def pick_rir(rir_root: Optional[str], cfg: MelConfig) -> Tuple[Optional[np.ndarray], str, str]:
    rf = pick_random_file(rir_root, exts=(".wav", ".flac"))
    if not rf:
        return None, "none", "none"
    try:
        r, sr = sf.read(rf, dtype="float32", always_2d=False)
        if isinstance(r, np.ndarray) and r.ndim > 1:
            r = r.mean(axis=1)
    except Exception:
        try:
            r, sr = _read_pcm_wav(rf)
        except Exception:
            return None, "none", "none"
    if sr != cfg.sr:
        r = librosa.resample(r, orig_sr=sr, target_sr=cfg.sr)
    EPS = 1e-9
    r = r / (float(np.max(np.abs(r)) + EPS))
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    t60_bucket = parse_t60_bucket_from_name(rf)
    mic_pos = "near" if "near" in rf.lower() else ("mid" if "mid" in rf.lower() else ("far" if "far" in rf.lower() else "unk"))
    return r.astype(np.float32), t60_bucket, mic_pos
def encode_labels(lb: LabelBinarizer, labs: List[str]) -> np.ndarray:
    """
    Ensure one-hot with shape (N, num_classes) even for binary labels.
    LabelBinarizer returns (N, 1) for binary; expand to (N, 2).
    """
    y = lb.transform(labs)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.shape[1] == 1:  # binary -> two columns
        y = np.hstack([1 - y, y])
    return y.astype(np.float32)


# --------------------- Data pipeline ------------------
class BatchGen(keras.utils.Sequence):
    def __init__(
        self,
        files,
        labels,
        lb,
        arm: ArmConfig,
        cfg: MelConfig,
        batch_size: int,
        noise_root: Optional[str],
        rir_root: Optional[str],
        specaug_cfg=(10, 6, 1, 1),
        p_specaugment=1.0,
        seed=0,
    ):
        self.files = files
        self.labels = labels
        self.lb = lb
        self.arm = arm
        self.cfg = cfg
        self.bs = batch_size
        self.noise_root = noise_root
        self.rir_root = rir_root
        self.specaug_cfg = specaug_cfg
        self.p_specaugment = p_specaugment
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return max(1, math.ceil(len(self.files) / self.bs))

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):
        slc = slice(idx * self.bs, min((idx + 1) * self.bs, len(self.files)))
        batch_files = self.files[slc]
        batch_labels = self.labels[slc]
        if len(batch_files) == 0:
            # return a dummy batch if something asks beyond bounds
            dummy = np.zeros((1, self.cfg.img_h, self.cfg.img_w, self.cfg.channels), np.float32)
            dummy_y = encode_labels(self.lb, [self.lb.classes_[0]])
            return dummy, dummy_y

        X, Y_labs = [], []
        for fp, lab in zip(batch_files, batch_labels):
            y = load_audio(fp, self.cfg)

            if self.arm.use_baseline:
                y = apply_baseline_ops(y, self.cfg)

            if self.arm.use_real_noise and random.random() < self.arm.p_noise:
                n = pick_noise(self.noise_root, self.cfg)
                if n is not None:
                    snr = random.choice(self.arm.snr_choices)
                    y = mix_at_snr(y, n, snr)

            if self.arm.use_rir and random.random() < self.arm.p_rir:
                rir, _, _ = pick_rir(self.rir_root, self.cfg)
                if rir is not None:
                    wet = float(np.random.uniform(0.7, 1.0))
                    y = convolve_rir(y, rir, wet)

            img = mel_image(y, self.cfg)
            if self.rng.rand() < self.p_specaugment:
                img = specaugment(img, *self.specaug_cfg)

            X.append(img)
            Y_labs.append(lab)

        X = np.stack(X, 0).astype(np.float32)
        Y = encode_labels(self.lb, Y_labs)
        return X, Y

# -------------------------- Model -------------------------------
def build_model(num_classes: int, cfg: MelConfig) -> keras.Model:
    inp = keras.Input(shape=(cfg.img_h, cfg.img_w, cfg.channels))
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(2)(x)
    x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(num_classes)(x)  # logits
    return keras.Model(inp, out)

def compile_model(model: keras.Model, cfg: TrainConfig):
    opt = keras.optimizers.Adam(learning_rate=cfg.lr)
    model.compile(
        optimizer=opt,
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )

# -------------------------- Metrics -----------------------------
def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    e = np.exp(logits)
    denom = e.sum(axis=1, keepdims=True) + 1e-9
    return (e / denom).astype(np.float32)

def compute_metrics(y_true_oh: np.ndarray, logits: np.ndarray) -> Dict:
    if len(y_true_oh) == 0:
        return {"macro_f1": 0.0, "balanced_acc": 0.0, "auprc": 0.0, "ece": 0.0, "brier": 0.0}

    probs = logits_to_probs(logits)
    y_true = np.argmax(y_true_oh, axis=1)
    y_pred = np.argmax(probs, axis=1)

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    ap = []
    for c in range(y_true_oh.shape[1]):
        pos = y_true_oh[:, c].sum()
        if (pos == 0) or (pos == len(y_true_oh)):
            continue
        ap.append(average_precision_score(y_true_oh[:, c], probs[:, c]))
    auprc = float(np.mean(ap)) if ap else 0.0

    conf = probs.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            m = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() == 0:
            continue
        acc_bin = correct[m].mean()
        conf_bin = conf[m].mean()
        ece += (m.sum() / len(conf)) * abs(acc_bin - conf_bin)

    brier = float(np.mean((y_true_oh - probs) ** 2))
    return {
        "macro_f1": float(macro_f1),
        "balanced_acc": float(bal_acc),
        "auprc": float(auprc),
        "ece": float(ece),
        "brier": float(brier),
    }

# -------------------------- Train/Eval ---------------------------
def train_one(model, train_gen, val_gen, tcfg: TrainConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=tcfg.patience, restore_best_weights=True),
        keras.callbacks.CSVLogger(os.path.join(out_dir, "training.csv")),
    ]
    total_steps = len(train_gen)
    if total_steps <= 0:
        raise ValueError("Train generator has zero batches.")

    requested = tcfg.steps_per_epoch or total_steps
    steps = max(1, min(requested, total_steps))

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=tcfg.epochs,
        steps_per_epoch=steps,
        callbacks=callbacks,
        verbose=2,
    )

def evaluate(model, files, labels, lb, cfg: MelConfig) -> Dict:
    logits, Y = [], []
    for fp, lab in zip(files, labels):
        y = load_audio(fp, cfg)
        img = mel_image(y, cfg)
        out = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        logits.append(out)
        Y.append(lab)
    logits = np.stack(logits) if logits else np.zeros((0, len(lb.classes_)), dtype=np.float32)
    Y = encode_labels(lb, Y) if Y else np.zeros((0, len(lb.classes_)), dtype=np.float32)
    return compute_metrics(Y, logits)

# ------------------------------ Main -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--id_dir", required=True, help="In-distribution dataset root (class folders).")
    p.add_argument("--ood_dir", required=True, help="OOD dataset root (class folders).")
    p.add_argument("--noise_dir", default=None, help="Directory with real noise wavs.")
    p.add_argument("--rir_dir", default=None, help="Directory with RIR wavs.")
    p.add_argument("--out", default="results", help="Output directory.")
    p.add_argument("--arms", nargs="+", default=["baseline", "noise_rir"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps_per_epoch", type=int, default=200, help="Fix equal steps across arms.")
    args = p.parse_args()

    if "noise_rir" in args.arms and (not args.noise_dir or not args.rir_dir):
        raise ValueError("noise_rir arm requires --noise_dir and --rir_dir")

    for d in [args.id_dir, args.ood_dir]:
        if not os.path.isdir(d):
            raise ValueError(f"Directory not found: {d}")

    mel_cfg = DEFAULT_MEL
    train_cfg = DEFAULT_TRAIN
    train_cfg.steps_per_epoch = args.steps_per_epoch

    # Load datasets (WAV/FLAC only)
    X_id, y_id = list_audio_files(args.id_dir)
    X_ood, y_ood = list_audio_files(args.ood_dir)

    # Class alignment across ID/OOD
    classes = sorted(set(y_id) | set(y_ood))
    if len(classes) < 2:
        raise ValueError(f"Need ≥2 classes, found {len(classes)}: {classes}")
    print(f"Classes ({len(classes)}):", classes)
    lb = LabelBinarizer().fit(classes)

    summary_rows = []
    for arm_name in args.arms:
        if arm_name not in ARMS:
            raise ValueError(f"Unknown arm: {arm_name}")
        arm = ARMS[arm_name]
        print(f"\n=== Arm: {arm.name} ===")
        arm_rows = []

        for seed in args.seeds:
            print(f"Seed {seed}")
            set_all_seeds(seed)

            # Fixed splits per seed
            X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_splits(
                X_id, y_id, seed=seed, test_size=0.2, val_size=0.2
            )

            # Generators
            train_gen = BatchGen(
                X_tr,
                y_tr,
                lb,
                arm,
                mel_cfg,
                batch_size=train_cfg.batch_size,
                noise_root=args.noise_dir,
                rir_root=args.rir_dir,
                specaug_cfg=(10, 6, 1, 1),
                p_specaugment=1.0,
                seed=seed,
            )
            val_gen = BatchGen(
                X_val,
                y_val,
                lb,
                arm,
                mel_cfg,
                batch_size=train_cfg.batch_size,
                noise_root=args.noise_dir,
                rir_root=args.rir_dir,
                specaug_cfg=(10, 6, 1, 1),
                p_specaugment=1.0,
                seed=seed,
            )

            # Model
            model = build_model(num_classes=len(classes), cfg=mel_cfg)
            compile_model(model, train_cfg)

            out_dir = os.path.join(args.out, arm.name, f"seed_{seed}")
            t0 = time.time()
            train_one(model, train_gen, val_gen, train_cfg, out_dir)
            wall = time.time() - t0

            # Evaluate
            m_id = evaluate(model, X_te, y_te, lb, mel_cfg)
            m_ood = evaluate(model, X_ood, y_ood, lb, mel_cfg)

            metrics = {
                "arm": arm.name,
                "seed": seed,
                "wall_time_s": float(wall),
                "id_macro_f1": m_id["macro_f1"],
                "id_bal_acc": m_id["balanced_acc"],
                "id_auprc": m_id["auprc"],
                "id_ece": m_id["ece"],
                "id_brier": m_id["brier"],
                "ood_macro_f1": m_ood["macro_f1"],
                "ood_bal_acc": m_ood["balanced_acc"],
                "ood_auprc": m_ood["auprc"],
                "ood_ece": m_ood["ece"],
                "ood_brier": m_ood["brier"],
            }
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            arm_rows.append(metrics)

        # Aggregate mean±std
        def agg(key):
            vals = np.array([r[key] for r in arm_rows], dtype=float)
            return float(vals.mean()), (float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)

        row = {
            "arm": arm.name,
            "id_macro_f1_mean": agg("id_macro_f1")[0],
            "id_macro_f1_std": agg("id_macro_f1")[1],
            "id_bal_acc_mean": agg("id_bal_acc")[0],
            "id_bal_acc_std": agg("id_bal_acc")[1],
            "id_auprc_mean": agg("id_auprc")[0],
            "id_auprc_std": agg("id_auprc")[1],
            "id_ece_mean": agg("id_ece")[0],
            "id_ece_std": agg("id_ece")[1],
            "id_brier_mean": agg("id_brier")[0],
            "id_brier_std": agg("id_brier")[1],
            "ood_macro_f1_mean": agg("ood_macro_f1")[0],
            "ood_macro_f1_std": agg("ood_macro_f1")[1],
            "ood_bal_acc_mean": agg("ood_bal_acc")[0],
            "ood_bal_acc_std": agg("ood_bal_acc")[1],
            "ood_auprc_mean": agg("ood_auprc")[0],
            "ood_auprc_std": agg("ood_auprc")[1],
            "ood_ece_mean": agg("ood_ece")[0],
            "ood_ece_std": agg("ood_ece")[1],
            "ood_brier_mean": agg("ood_brier")[0],
            "ood_brier_std": agg("ood_brier")[1],
        }
        summary_rows.append(row)

    # Save summary
    os.makedirs(args.out, exist_ok=True)
    import csv

    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Paired t-test (if multiple seeds present)
    def load_arm_seeds(arm):
        vals = []
        for seed in args.seeds:
            pth = os.path.join(args.out, arm, f"seed_{seed}", "metrics.json")
            if os.path.isfile(pth):
                with open(pth) as f:
                    vals.append(json.load(f)["ood_macro_f1"])
        return np.array(vals, float)

    if "baseline" in args.arms and "noise_rir" in args.arms and len(args.seeds) >= 2:
        a = load_arm_seeds("baseline")
        b = load_arm_seeds("noise_rir")
        if len(a) == len(b) and len(a) >= 2:
            t, pval = ttest_rel(b, a)
            with open(os.path.join(args.out, "significance.txt"), "w") as f:
                f.write(f"Paired t-test OOD macro-F1: noise_rir vs baseline: t={t:.3f}, p={pval:.5f}\n")

    print("\nDone. See:", args.out)

if __name__ == "__main__":
    main()
