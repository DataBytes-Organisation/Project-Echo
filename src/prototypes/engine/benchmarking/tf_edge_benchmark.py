# %%
# Written by Tharrun Satish - s225565588

import os
import time
import statistics
from pathlib import Path
import tensorflow as tf #Comment this line for Edge Deployement
import numpy as np
import librosa

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# %%
""" 
For the purposes of running the model on RasberryPi(Or any Edge Deployement Scenario) Uncomment the Following Line code block and Comment tensorflow Import.
"""
#The Following need to be uncommented in the scenario of Edge Deployement
""" 
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter
"""
# %%

"""
Enter Model Path Here, This script is designed for EfficientNetV2. 

Incase of Model Replacement, Sections that require change will be highlighted later.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
tflite_model_path = str(SCRIPT_DIR / "Models_For_Benchmarking/efficientnetv2_project_echo.tflite")

"""
The following code block might require change in tandem with the model being used. This is designed for the use of EfficientNetV2.
Change config based on model input details.
"""

# --- Preprocessing config (VERIFY against training pipeline) ---
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
N_FRAMES = 313          # fixed by the model's input shape
CLIP_DURATION_S = (N_FRAMES * HOP_LENGTH) / SAMPLE_RATE 

# %%
"""" 
Loads the Tflite model and gets Interpreter details.
Load time and model size are captured here as benchmark metrics.
"""

model_size_mb = Path(tflite_model_path).stat().st_size / (1024 * 1024)

t0 = time.perf_counter()
interpreter = tf.lite.Interpreter(model_path=tflite_model_path) #Comment this line for Edge Deployement. This is for development Environment
#interpreter = Interpreter(model_path=tflite_model_path, num_threads=4) #Use this line when scenario is Edge Deployement
interpreter.allocate_tensors()
load_time_ms = (time.perf_counter() - t0) * 1000

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input details:", input_details)
print("Output details:", output_details)
print(f"\nModel size: {model_size_mb:.3f} MB")
print(f"Load time: {load_time_ms:.2f} ms")
print(f"Implied clip duration for {N_FRAMES} frames @ hop={HOP_LENGTH}, sr={SAMPLE_RATE}: "
      f"{CLIP_DURATION_S:.3f} s")

# %%
""" 
Audio file used for testing. Change pathway to actual test data.
"""

test_data_path = str(SCRIPT_DIR/ "../../../../models-and-data/samples/store_audio/Alauda Arvensis.wav")

# %%
def load_and_pad_waveform(path, sr, target_duration_s):
    """Load audio, resample to sr, and pad/trim to a fixed duration so the
    resulting spectrogram lands on exactly N_FRAMES time steps."""
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(round(target_duration_s * sr))

    if len(waveform) < target_len:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]

    return waveform.astype(np.float32)

#This function is coded for EfficietNetV2, Require change based on model input.
def waveform_to_logmel(waveform, sr, n_fft, hop_length, n_mels, n_frames):
    """Compute a log-mel spectrogram and force it to exactly n_frames columns
    (pad with silence-floor or trim), matching the model's fixed input shape."""
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Force exact frame count
    if log_mel.shape[1] < n_frames:
        pad_width = n_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=log_mel.min())
    elif log_mel.shape[1] > n_frames:
        log_mel = log_mel[:, :n_frames]

    return log_mel


# %%
waveform = load_and_pad_waveform(test_data_path, SAMPLE_RATE, CLIP_DURATION_S)
log_mel = waveform_to_logmel(waveform, SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, N_FRAMES)

# Reshape to model's expected [batch, channel, mel_bins, time_frames]
model_input = log_mel[np.newaxis, np.newaxis, :, :].astype(np.float32)
print("Prepared input shape:", model_input.shape)  

# %%

"Sanity check might require change in cases where the model is changed."
input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

# %%
""" 
Single sanity-check inference (not timed) to confirm the model runs correctly
on this audio before benchmarking.

"""

interpreter.set_tensor(input_index, model_input)
interpreter.invoke()

class_scores = interpreter.get_tensor(output_index)  

print("Class scores shape:", class_scores.shape)
top_class = int(np.argmax(class_scores, axis=-1)[0])
print("Top class index:", top_class, "score:", float(class_scores[0, top_class]))
print("Sum:", class_scores.sum())
print("Min/Max:", class_scores.min(), class_scores.max())

# %%
""" 
BENCHMARKING: inference time, memory use, prediction consistency
Runs repeated inference on the SAME prepared spectrogram input so timing
and consistency results are directly comparable.
"""

BENCH_RUNS = 50
BENCH_WARMUP = 5


def percentile(values, p):
    values = sorted(values)
    k = (len(values) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def run_inference_benchmark(interpreter, input_index, model_input, output_details, runs, warmup):
    for _ in range(warmup):
        interpreter.set_tensor(input_index, model_input)
        interpreter.invoke()

    proc = psutil.Process(os.getpid()) if _HAS_PSUTIL else None
    peak_rss = proc.memory_info().rss if proc else 0

    latencies_ms = []
    output_snapshots = []

    for _ in range(runs):
        interpreter.set_tensor(input_index, model_input)
        t0 = time.perf_counter()
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        if proc:
            peak_rss = max(peak_rss, proc.memory_info().rss)

        outs = [interpreter.get_tensor(d["index"]).copy() for d in output_details]
        output_snapshots.append(outs)

    return latencies_ms, output_snapshots, (peak_rss / (1024 * 1024) if proc else None)


def summarize_latency(latencies_ms):
    total_s = sum(latencies_ms) / 1000
    return {
        "runs": len(latencies_ms),
        "avg_ms": round(statistics.mean(latencies_ms), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "std_ms": round(statistics.pstdev(latencies_ms), 3) if len(latencies_ms) > 1 else 0.0,
        "p50_ms": round(percentile(latencies_ms, 50), 3),
        "p90_ms": round(percentile(latencies_ms, 90), 3),
        "p99_ms": round(percentile(latencies_ms, 99), 3),
        "throughput_fps": round(len(latencies_ms) / total_s, 2) if total_s > 0 else 0.0,
    }


def check_prediction_consistency(output_snapshots, output_details):
    ref = output_snapshots[0]
    per_output = []

    for i, d in enumerate(output_details):
        max_abs_diff = 0.0
        bit_identical = True
        for outs in output_snapshots[1:]:
            ref_out, out = ref[i], outs[i]
            if not np.array_equal(ref_out, out):
                bit_identical = False
            diff = np.max(np.abs(ref_out.astype(np.float64) - out.astype(np.float64)))
            max_abs_diff = max(max_abs_diff, float(diff))
        per_output.append({
            "output_name": d["name"],
            "bit_identical_across_runs": bit_identical,
            "max_abs_output_diff": round(max_abs_diff, 8),
        })
    return per_output


print(f"Running {BENCH_WARMUP} warmup + {BENCH_RUNS} timed inferences on the loaded audio...")
latencies_ms, output_snapshots, peak_rss_mb = run_inference_benchmark(
    interpreter, input_index, model_input, output_details, BENCH_RUNS, BENCH_WARMUP
)

latency_stats = summarize_latency(latencies_ms)
consistency_stats = check_prediction_consistency(output_snapshots, output_details)

benchmark_report = {
    "model": tflite_model_path,
    "audio_file": test_data_path,
    "clip_duration_s": round(CLIP_DURATION_S, 3),
    "model_size_mb": round(model_size_mb, 3),
    "load_time_ms": round(load_time_ms, 3),
    "peak_rss_mb": round(peak_rss_mb, 2) if peak_rss_mb is not None else "psutil not installed",
    **latency_stats,
    "prediction_consistency": consistency_stats,
}

# %%
print("Benchmark Report")
for k, v in benchmark_report.items():
    print(f"{k:>22}: {v}")
