import argparse
import librosa
from pathlib import Path
import sys
import math
import numpy as np
import tensorflow as tf
from scipy import signal
from omegaconf import OmegaConf

def get_scale_factor(cfg):
	"""
	Determines the scaling factor 's' based on the training configuration.
	"""
	metric_loss_type = cfg.training.get("use_arcface", None)

	scale_factor = 1.0

	if metric_loss_type == "arcface":
		scale_factor = cfg.training.arcface.s
		print(f"  [i] Detected ArcFace training. Setting scale factor s={scale_factor}")
	elif metric_loss_type == "circle":
		scale_factor = cfg.training.circle.s
		print(f"  [i] Detected CircleLoss training. Setting scale factor s={scale_factor}")
	else:
		print("  [i] No metric loss detected (Standard Linear). Scale factor s=1.0")
		scale_factor = 1.0

	return scale_factor

def preprocess_with_librosa(waveform: np.ndarray, pp_data: dict) -> np.ndarray:
	"""
	A torch-free preprocessor for audio data that uses a dictionary of
	pre-computed data (including filter bank).
	"""
	# Extract parameters from the loaded dictionary
	sample_rate = pp_data["sample_rate"]
	target_duration_samples = int(pp_data["audio_clip_duration"] * sample_rate)
	n_fft = pp_data["n_fft"]
	hop_length = pp_data["hop_length"]
	mel_filters = pp_data["mel_filters"]
	top_db = pp_data["top_db"]

	# Pad or truncate to target duration
	current_len = waveform.shape[-1]
	if current_len > target_duration_samples:
		waveform = waveform[..., :target_duration_samples]
	elif current_len < target_duration_samples:
		if current_len > 0:
			repeats = int(math.ceil(target_duration_samples / current_len))
			waveform = np.tile(waveform, repeats)
		waveform = waveform[..., :target_duration_samples]

	# Manually pad for STFT consistency
	padding = n_fft // 2
	waveform = np.pad(waveform, padding, mode="reflect")

	# Calculate STFT Power Spectrogram using librosa
	hann_window_periodic_np = signal.get_window("hann", n_fft, fftbins=True)
	stft_complex = librosa.stft(
		y=waveform,
		n_fft=n_fft,
		hop_length=hop_length,
		center=False,
		window=hann_window_periodic_np,
	)
	stft_power = np.abs(stft_complex) ** 2

	# Apply the pre-computed torchaudio filter bank
	mel_spectrogram = (stft_power.T @ mel_filters).T

	# Convert to dB scale
	melspec_db = librosa.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=top_db)

	# Normalize
	normalized_spec = (melspec_db + top_db) / top_db

	# Add a channel and batch dimension for TFLite (NHWC)
	normalized_spec = normalized_spec[..., np.newaxis]
	return normalized_spec[np.newaxis, ...]


def main(args):
	"""
	Main function to run inference on a directory of audio files using a TFLite
	model and assets from a trained model directory.
	"""
	model_dir = Path(args.model_dir)
	audio_dir = Path(args.audio_dir)
	threshold = args.threshold

	print(f"Loading assets from '{model_dir}'...")
	config_path = model_dir / ".hydra" / "config.yaml"
	if not config_path.exists():
		print(f"Error: Could not find configuration file at '{config_path}'.")
		sys.exit(1)

	cfg = OmegaConf.load(config_path)
	scale_factor = get_scale_factor(cfg)
	print(f"  [i] Loaded configuration for model '{cfg.model.name}'.")

	tflite_model_path = model_dir / f"{cfg.model.name}.tflite"
	preprocess_path = model_dir / "preprocess.npy"
	class_names_path = model_dir / "class_names.txt"

	required_files = {
		"TFLite Model": tflite_model_path,
		"Class Names": class_names_path,
		"Preprocessing Data": preprocess_path,
	}
	for name, path in required_files.items():
		if not path.exists():
			print(f"Error: Missing required file: {name} at '{path}'")
			sys.exit(1)

	try:
		interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
		interpreter.allocate_tensors()
		with open(class_names_path, "r") as f:
			class_names = [line.strip() for line in f.readlines() if line.strip()]
		pp_data = np.load(preprocess_path, allow_pickle=True).item()
		print("Assets loaded successfully.")
	except Exception as e:
		print(f"Error loading assets: {e}")
		sys.exit(1)

	if not audio_dir.is_dir():
		print(f"Error: Provided audio path '{audio_dir}' is not a directory.")
		sys.exit(1)

	print(f"\nSearching for audio files in '{audio_dir}'...")
	audio_extensions = [".wav", ".mp3", ".flac", ".ogg"]
	audio_files = []
	for ext in audio_extensions:
		audio_files.extend(list(audio_dir.rglob(f"*{ext}")))
		audio_files.extend(list(audio_dir.rglob(f"*{ext.upper()}")))

	audio_files = sorted(list(set(audio_files)))

	if not audio_files:
		print(f"No audio files found in '{audio_dir}'.")
		sys.exit(1)

	print(f"Found {len(audio_files)} audio files to process.")

	# Process Each Audio File
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]
	clip_duration = pp_data["audio_clip_duration"]
	target_sr = pp_data["sample_rate"]

	# Get target width from the model's input details.
	target_width = input_details["shape"][2]  # Shape is (N,H,W,C)
	print(f"\n[i] Model expects input width: {target_width}. Spectrograms will be resized if necessary.")

	try:
		for audio_file in audio_files:
			print(f"\n----- Processing: {audio_file.relative_to(audio_dir)} -----")
			try:
				waveform, _ = librosa.load(audio_file, sr=target_sr, mono=True)
			except Exception as e:
				print(f"  [!] Error loading audio file: {e}")
				continue

			chunk_size_samples = int(clip_duration * target_sr)
			total_samples = waveform.shape[-1]
			current_pos_samples = 0

			while current_pos_samples < total_samples:
				start_time_s = current_pos_samples / target_sr
				chunk = waveform[current_pos_samples : current_pos_samples + chunk_size_samples]

				if len(chunk) == 0:
					break

				spectrogram = preprocess_with_librosa(chunk, pp_data)

				# Ensure spectrogram width matches the model's expected input width.
				current_width = spectrogram.shape[2]
				if current_width != target_width:
					if current_width > target_width:
						spectrogram = spectrogram[:, :, :target_width, :]
					else:  # current_width < target_width
						pad_width = target_width - current_width
						paddings = ((0, 0), (0, 0), (0, pad_width), (0, 0))
						spectrogram = np.pad(spectrogram, paddings, mode="constant", constant_values=0)

				interpreter.set_tensor(input_details["index"], spectrogram)
				interpreter.invoke()
				logits = interpreter.get_tensor(output_details["index"])
				logits = logits * scale_factor

				probabilities = tf.nn.softmax(logits[0]).numpy()
				top_prob = np.max(probabilities)
				pred_idx = np.argmax(probabilities)

				if top_prob >= threshold:
					prediction_label = class_names[pred_idx]
					confidence_str = f"Confidence: {top_prob:.2%}"
				else:
					prediction_label = "unknown"
					confidence_str = f"Highest confidence: {top_prob:.2%}"

				print(f"  [{start_time_s:06.2f}s] Prediction: '{prediction_label}' ({confidence_str})")
				current_pos_samples += chunk_size_samples

	except KeyboardInterrupt:
		print("\nProcessing stopped by user.")
	finally:
		print("\n--- All files processed ---")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Run inference on a directory of audio files using a TFLite model.")
	parser.add_argument(
		"--model_dir",
		type=str,
		required=True,
		help="Path to the trained model directory containing artifacts (TFLite, preprocess.npy, etc.).",
	)
	parser.add_argument("--audio_dir", type=str, required=True, help="Path to the directory of audio files to process.")
	parser.add_argument(
		"--threshold", type=float, default=0.01, help="Confidence threshold for predictions (from 0.0 to 1.0)."
	)
	args = parser.parse_args()
	main(args)
