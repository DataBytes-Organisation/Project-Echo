import argparse
import torch
import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf
from pathlib import Path
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import defaultdict
import librosa
from scipy import signal

# Add project root to path to allow importing 'model' and 'dataset'
sys.path.append(str(Path(__file__).resolve().parent))
from model import Model
from dataset import SpectrogramDataset, index_directory, validation_collate_fn


def preprocess_with_librosa(waveform: np.ndarray, pp_data: dict) -> np.ndarray:
	"""
	A torch-free preprocessor for audio data that uses a dictionary of
	pre-computed data (including filter bank).
	"""
	# Extract parameters from the loaded dictionary
	sample_rate = pp_data['sample_rate']
	target_duration_samples = int(pp_data['audio_clip_duration'] * sample_rate)
	n_fft = pp_data['n_fft']
	hop_length = pp_data['hop_length']
	mel_filters = pp_data['mel_filters']
	top_db = pp_data['top_db']

	# Pad or truncate to target duration
	current_len = waveform.shape[-1]
	if current_len > target_duration_samples:
		waveform = waveform[..., :target_duration_samples]
	elif current_len < target_duration_samples:
		pad_amount = target_duration_samples - current_len
		waveform = np.pad(waveform, (0, pad_amount))

	# Manually pad for STFT consistency
	padding = n_fft // 2
	waveform = np.pad(waveform, padding, mode='reflect')

	# Calculate STFT Power Spectrogram using librosa
	hann_window_periodic_np = signal.get_window('hann', n_fft, fftbins=True)
	stft_complex = librosa.stft(
		y=waveform,
		n_fft=n_fft,
		hop_length=hop_length,
		center=False,
		window=hann_window_periodic_np,
	)
	stft_power = np.abs(stft_complex)**2

	# Apply the pre-computed torchaudio filter bank
	mel_spectrogram = (stft_power.T @ mel_filters).T

	# Convert to dB scale
	melspec_db = librosa.power_to_db(
		mel_spectrogram,
		ref=1.0,
		amin=1e-10,
		top_db=top_db
	)

	# Normalize
	normalized_spec = (melspec_db + top_db) / top_db

	# Add a channel and batch dimension for TFLite (NHWC)
	normalized_spec = normalized_spec[..., np.newaxis]
	return normalized_spec[np.newaxis, ...]


def run_pytorch_inference(model: Model, input_data: torch.Tensor) -> np.ndarray:
	"""Runs inference on the PyTorch model."""
	with torch.no_grad():
		output = model(input_data)
	return output.numpy()

def run_tflite_inference(interpreter: tf.lite.Interpreter, input_data: np.ndarray) -> np.ndarray:
	"""Runs inference on the TFLite model."""
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]

	interpreter.set_tensor(input_details['index'], input_data)
	interpreter.invoke()

	return interpreter.get_tensor(output_details['index'])

def compare_models(args):
	"""
	Loads a PyTorch and TFLite model from a specified directory,
	runs inference on dummy data, and compares their outputs.
	"""
	model_dir = Path(args.model_dir)

	print("=" * 70)
	print("PyTorch vs. TFLite End-to-End Validation")
	print("=" * 70)

	# Load Config and Find Models
	print("\n--- Step 1: Loading configuration and finding models ---")
	config_path = model_dir / ".hydra" / "config.yaml"
	if not config_path.exists():
		print(f"Error: Could not find configuration file at '{config_path}'.")
		sys.exit(1)

	cfg = OmegaConf.load(config_path)

	pytorch_model_path = model_dir / f"best_{cfg.model.name}.pth"
	tflite_model_path = model_dir / f"{cfg.model.name}.tflite"

	if not pytorch_model_path.exists():
		print(f"Error: PyTorch model not found at '{pytorch_model_path}'")
		sys.exit(1)
	if not tflite_model_path.exists():
		print(f"Error: TFLite model not found at '{tflite_model_path}'")
		sys.exit(1)

	print(f"  [✓] PyTorch model: {pytorch_model_path.name}")
	print(f"  [✓] TFLite model:  {tflite_model_path.name}")

	# Load Models & Assets
	print("\n--- Step 2: Loading models and assets ---")
	try:
		# Load PyTorch model
		class_names_path = model_dir / "class_names.txt"
		with open(class_names_path, "r") as f:
			num_classes = len([line for line in f if line.strip()])

		OmegaConf.set_struct(cfg, False)
		cfg.data.num_classes = num_classes
		OmegaConf.set_struct(cfg, True)

		pytorch_model = Model(cfg)
		checkpoint = torch.load(pytorch_model_path, map_location='cpu')
		state_dict = checkpoint.get('model_state_dict', checkpoint)
		pytorch_model.load_state_dict(state_dict)
		pytorch_model.eval()
		print("  [✓] PyTorch model loaded.")

		# Load TFLite model
		interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
		interpreter.allocate_tensors()
		print("  [✓] TFLite model loaded.")

		# Load preprocessing data for Librosa pipeline
		preprocess_path = model_dir / "preprocess.npy"
		if not preprocess_path.exists():
			print(f"Error: Preprocessing data not found at '{preprocess_path}'")
			sys.exit(1)
		pp_data = np.load(preprocess_path, allow_pickle=True).item()
		print("  [✓] Librosa preprocessing data loaded.")

	except Exception as e:
		print(f"Error loading models: {e}")
		sys.exit(1)

	print("\n--- Step 2a: Loading validation dataset ---")
	input_details = interpreter.get_input_details()[0]
	target_width = input_details['shape'][2]
	print(f"  [i] TFLite model expects input width: {target_width}")

	audio_files, labels, class_names = index_directory(cfg.system.audio_data_directory)
	torch.manual_seed(cfg.training.seed)

	class_indices = defaultdict(list)
	for idx, label in enumerate(labels):
		class_indices[label].append(idx)

	val_indices = []
	for class_label, indices_list in class_indices.items():
		indices = torch.tensor(indices_list)
		shuffled_class_indices = indices[torch.randperm(len(indices))]
		n_val = int(len(indices) * cfg.data.val_split)
		val_indices.extend(shuffled_class_indices[:n_val].tolist())

	print(f"  [✓] Found {len(val_indices)} validation samples.")

	# This dataset is for the PyTorch (torchaudio) pipeline
	val_dataset_torch = SpectrogramDataset(
		[audio_files[i] for i in val_indices],
		[labels[i] for i in val_indices],
		cfg,
		is_train=False,
		target_width=target_width
	)
	val_loader_torch = torch.utils.data.DataLoader(
		val_dataset_torch, batch_size=1, shuffle=False,
		num_workers=os.cpu_count(), collate_fn=validation_collate_fn
	)

	# Run Comparisons
	print(f"\n--- Step 3: Running comparisons on validation set ({len(val_indices)} samples) ---")

	pytorch_file_preds = []
	tflite_file_preds = []
	true_file_labels = []
	pytorch_class_confidences = defaultdict(list)
	tflite_class_confidences = defaultdict(list)

	val_audio_paths = [audio_files[i] for i in val_indices]

	iterator = zip(val_loader_torch, val_audio_paths)

	for i, ((torch_inputs, torch_labels), audio_path) in enumerate(tqdm(iterator, total=len(val_audio_paths), desc="Comparing pipelines")):

		# For the first sample, explicitly verify the spectrograms are consistent
		if i == 0:
			print("\n--- Verifying librosa preprocessing against torchaudio ---")
			torchaudio_spec = torch_inputs[0].to('cpu').numpy()

			target_sr = pp_data['sample_rate']
			clip_duration = pp_data['audio_clip_duration']
			chunk_size_samples = int(clip_duration * target_sr)
			waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)
			chunk_waveform = waveform[0:chunk_size_samples]

			librosa_spec = preprocess_with_librosa(chunk_waveform, pp_data)

			current_width = librosa_spec.shape[2]
			if current_width != target_width:
				if current_width > target_width:
					librosa_spec = librosa_spec[:, :, :target_width, :]
				else:
					pad_width = target_width - current_width
					paddings = ((0, 0), (0, 0), (0, pad_width), (0, 0))
					librosa_spec = np.pad(librosa_spec, paddings, mode='constant', constant_values=0)

			librosa_spec_reshaped = librosa_spec.squeeze(axis=(0, 3))[np.newaxis, :, :]

			spec_diff = np.abs(torchaudio_spec - librosa_spec_reshaped)
			max_spec_diff = np.max(spec_diff)

			print(f"  Max element-wise difference between spectrograms: {max_spec_diff:.8f}")
			spec_tolerance = 1e-5
			if max_spec_diff < spec_tolerance:
				print("  [✓] PASSED: Librosa-based preprocessing is consistent with torchaudio.")
			else:
				print("  [✗] FAILED: Librosa-based preprocessing differs significantly from torchaudio.")

		true_label_idx = torch_labels[0].item()
		true_file_labels.append(true_label_idx)

		# --- PyTorch Pipeline (torchaudio) ---
		pytorch_outputs_chunks = run_pytorch_inference(pytorch_model, torch_inputs.to('cpu'))
		pytorch_output_agg = np.mean(pytorch_outputs_chunks, axis=0)
		pytorch_probs = torch.nn.functional.softmax(torch.from_numpy(pytorch_output_agg), dim=0).numpy()

		pytorch_file_preds.append(np.argmax(pytorch_probs))
		pytorch_class_confidences[true_label_idx].append(np.max(pytorch_probs))

		# --- TFLite Pipeline (librosa) ---
		target_sr = pp_data['sample_rate']
		clip_duration = pp_data['audio_clip_duration']
		chunk_size_samples = int(clip_duration * target_sr)

		waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)

		tflite_outputs_chunks = []
		current_pos_samples = 0
		while current_pos_samples < waveform.shape[-1]:
			chunk_waveform = waveform[current_pos_samples : current_pos_samples + chunk_size_samples]

			if len(chunk_waveform) == 0:
				break

			spectrogram = preprocess_with_librosa(chunk_waveform, pp_data)

			current_width = spectrogram.shape[2]
			if current_width != target_width:
				if current_width > target_width:
					spectrogram = spectrogram[:, :, :target_width, :]
				else:
					pad_width = target_width - current_width
					paddings = ((0, 0), (0, 0), (0, pad_width), (0, 0))
					spectrogram = np.pad(spectrogram, paddings, mode='constant', constant_values=0)

			tflite_chunk_output = run_tflite_inference(interpreter, spectrogram)
			tflite_outputs_chunks.append(tflite_chunk_output)
			current_pos_samples += chunk_size_samples

		tflite_output_agg = np.mean(np.vstack(tflite_outputs_chunks), axis=0)
		tflite_probs = tf.nn.softmax(tflite_output_agg).numpy()

		tflite_file_preds.append(np.argmax(tflite_probs))
		tflite_class_confidences[true_label_idx].append(np.max(tflite_probs))

	print("\n" + "=" * 70)
	print("Validation Summary")
	print("=" * 70)

	print(f"  Number of samples validated: {len(val_indices)}")

	print("\n--- Classification Accuracy ---")
	pytorch_accuracy = accuracy_score(true_file_labels, pytorch_file_preds)
	tflite_accuracy = accuracy_score(true_file_labels, tflite_file_preds)

	print(f"  PyTorch Model Accuracy (torchaudio): {pytorch_accuracy:.4f}")
	print(f"  TFLite Model Accuracy (librosa):   {tflite_accuracy:.4f}")

	if np.isclose(pytorch_accuracy, tflite_accuracy, atol=0.01):
		print("\n  [✓] PASSED: TFLite (librosa) pipeline accuracy is consistent with PyTorch (torchaudio) pipeline.")
	else:
		print("\n  [✗] FAILED: TFLite (librosa) pipeline accuracy differs significantly from PyTorch (torchaudio).")

	# --- Per-Class Accuracy Analysis ---
	pytorch_avg_confidences = {}
	for class_idx, conf_list in pytorch_class_confidences.items():
		if conf_list:
			pytorch_avg_confidences[class_names[class_idx]] = np.mean(conf_list)

	tflite_avg_confidences = {}
	for class_idx, conf_list in tflite_class_confidences.items():
		if conf_list:
			tflite_avg_confidences[class_names[class_idx]] = np.mean(conf_list)

	print("\n--- Per-Class Analysis (based on TFLite predictions) ---")
	cm = confusion_matrix(true_file_labels, tflite_file_preds, labels=np.arange(len(class_names)))
	with np.errstate(divide='ignore', invalid='ignore'):
		per_class_acc = cm.diagonal() / cm.sum(axis=1)

	class_accuracies = []
	for i, class_name in enumerate(class_names):
		if cm.sum(axis=1)[i] > 0:
			class_accuracies.append((class_name, per_class_acc[i]))

	sorted_accuracies = sorted(class_accuracies, key=lambda item: item[1], reverse=True)

	if not sorted_accuracies:
		print("  Could not calculate per-class accuracies (no validation samples found).")
	else:
		num_to_show = min(10, len(sorted_accuracies))

		print(f"\n  Top {num_to_show} Best Performing Classes:")
		header = f"    {'Class Name':<30} {'Accuracy':<12} {'PT Avg Conf':<15} {'TFLite Avg Conf'}"
		print(header)
		print(f"    {'-'*30} {'-'*12} {'-'*15} {'-'*15}")
		for class_name, acc in sorted_accuracies[:num_to_show]:
			acc_str = f"{acc:.2%}" if not np.isnan(acc) else "N/A"
			pt_conf_str = f"{pytorch_avg_confidences.get(class_name, 0):.2%}"
			tflite_conf_str = f"{tflite_avg_confidences.get(class_name, 0):.2%}"
			print(f"    - {class_name:<28} {acc_str:<12} {pt_conf_str:<15} {tflite_conf_str}")

		print(f"\n  Top {num_to_show} Worst Performing Classes:")
		print(header)
		print(f"    {'-'*30} {'-'*12} {'-'*15} {'-'*15}")
		worst_classes = sorted_accuracies[-num_to_show:]
		for class_name, acc in reversed(worst_classes):
			acc_str = f"{acc:.2%}" if not np.isnan(acc) else "N/A"
			pt_conf_str = f"{pytorch_avg_confidences.get(class_name, 0):.2%}"
			tflite_conf_str = f"{tflite_avg_confidences.get(class_name, 0):.2%}"
			print(f"    - {class_name:<28} {acc_str:<12} {pt_conf_str:<15} {tflite_conf_str}")

	print("=" * 70)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Validate a TFLite model against its original PyTorch checkpoint."
	)
	parser.add_argument(
		"model_dir",
		type=str,
		help="Path to the trained model directory containing all artifacts."
	)
	parser.add_argument(
		"--num_tests", # This argument will now be ignored/deprecated
		type=int,
		default=0, # Set default to 0 as it's not used for validation set size
		help="This argument is deprecated. The validation set size will be used."
	)
	args = parser.parse_args()
	compare_models(args)
