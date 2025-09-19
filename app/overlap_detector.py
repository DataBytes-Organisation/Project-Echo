# app/overlap_detector.py
import io
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import medfilt

SR = 16000
N_MELS = 64
N_FFT = 1024
HOP = 160   # 10ms
WIN = 400   # 25ms
BANDS = [(0, 250), (250, 2000), (2000, 8000)]  # 低/中/高 频段(Hz)
BAND_NAMES = ["low", "mid", "high"]

def load_audio_from_bytes(b: bytes, sr=SR):
    data, file_sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if file_sr != sr:
        data = librosa.resample(y=data, orig_sr=file_sr, target_sr=sr)
    return data

def logmel(y, sr=SR):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, win_length=WIN, n_mels=N_MELS, power=2.0
    )
    S_db = librosa.power_to_db(S + 1e-10)
    return S_db  # [n_mels, T]

def band_flux(S_db, sr=SR):
    """按频段计算谱通量(帧间差分>0)，作为“事件活跃度”基线。"""
    n_mels, T = S_db.shape
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr//2)
    # 计算每帧对前一帧的正增量（谱通量）
    d = np.maximum(0.0, np.diff(S_db, axis=1))
    d = np.pad(d, ((0,0),(1,0)))  # 对齐到T
    band_fluxes = []
    for (lo, hi) in BANDS:
        mask = (mel_freqs >= lo) & (mel_freqs < hi)
        # 频段内取均值
        bf = d[mask].mean(axis=0) if np.any(mask) else np.zeros(T, dtype=np.float32)
        band_fluxes.append(bf)
    band_fluxes = np.stack(band_fluxes, axis=1)  # [T, B]
    # 中值滤波去毛刺
    band_fluxes = medfilt(band_fluxes, kernel_size=(5,1))
    # 0-1 归一化（逐频段）
    eps = 1e-6
    mn = band_fluxes.min(axis=0, keepdims=True)
    mx = band_fluxes.max(axis=0, keepdims=True)
    norm = (band_fluxes - mn) / (mx - mn + eps)
    return norm  # [T, B]

def detect_overlaps(y, sr=SR, th=0.45, min_dur=0.15):
    """
    返回可能的“重叠事件”：
    - 规则：同一帧内，≥2个频段活跃视为重叠
    - 输出：事件的起止时间、活跃频段标签、置信度（该段内的平均活跃度）
    """
    S_db = logmel(y, sr)
    bf = band_flux(S_db, sr)  # [T, B]
    active = (bf >= th).astype(np.int32)       # [T, B]
    multi = active.sum(axis=1) >= 2            # [T] 是否重叠
    events = []
    T = bf.shape[0]
    i = 0
    frame_sec = HOP / sr
    while i < T:
        if multi[i]:
            j = i
            while j < T and multi[j]:
                j += 1
            dur = (j - i) * frame_sec
            if dur >= min_dur:
                seg = bf[i:j]                      # [L, B]
                mean_flux = seg.mean(axis=0)       # [B]
                hot_bands = [BAND_NAMES[k] for k,v in enumerate(mean_flux) if v >= th]
                conf = float(mean_flux.mean())
                events.append({
                    "labels": hot_bands if hot_bands else ["overlap"],
                    "start": round(i * frame_sec, 3),
                    "end": round(j * frame_sec, 3),
                    "confidence": round(conf, 3)
                })
            i = j
        else:
            i += 1
    # 统计信息，便于HMI可视化
    meta = {
        "fps": round(1.0 / frame_sec, 2),
        "frames": T,
        "bands": BAND_NAMES,
        "threshold": th
    }
    return events, meta

def run_from_bytes(b: bytes, th=0.45, min_dur=0.15):
    y = load_audio_from_bytes(b, sr=SR)
    return detect_overlaps(y, sr=SR, th=th, min_dur=min_dur)
