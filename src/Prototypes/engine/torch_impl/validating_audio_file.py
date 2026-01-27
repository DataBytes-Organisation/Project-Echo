import gradio as gr
import torch
import torchaudio
from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
import sys
import tensorflow as tf
import argparse
import librosa
from scipy import signal
import math
import pandas as pd  # Added: Required for gr.BarPlot
from model import Model

# --- Load Assets on Startup ---
ASSETS = {
	"tflite_interpreter": None,
	"pytorch_model": None,
	"cfg": None,
	"class_names": [],
	"target_width": None,
	"pp_data": None,
	"input_details": None,
	"output_details": None,
	"error": None,
}


def preprocess_with_librosa(waveform: np.ndarray, pp_data: dict) -> np.ndarray:
	"""
	A torch-free preprocessor for audio data that uses a dictionary of
	pre-computed data (including filter bank).
	"""
	sample_rate = pp_data["sample_rate"]
	target_duration_samples = int(pp_data["audio_clip_duration"] * sample_rate)
	n_fft = pp_data["n_fft"]
	hop_length = pp_data["hop_length"]
	mel_filters = pp_data["mel_filters"]
	top_db = pp_data["top_db"]

	current_len = waveform.shape[-1]
	if current_len > target_duration_samples:
		waveform = waveform[..., :target_duration_samples]
	elif current_len < target_duration_samples:
		if current_len > 0:
			repeats = int(math.ceil(target_duration_samples / current_len))
			waveform = np.tile(waveform, repeats)
		waveform = waveform[..., :target_duration_samples]

	padding = n_fft // 2
	waveform = np.pad(waveform, padding, mode="reflect")

	hann_window_periodic_np = signal.get_window("hann", n_fft, fftbins=True)
	stft_complex = librosa.stft(
		y=waveform,
		n_fft=n_fft,
		hop_length=hop_length,
		center=False,
		window=hann_window_periodic_np,
	)
	stft_power = np.abs(stft_complex) ** 2
	mel_spectrogram = (stft_power.T @ mel_filters).T

	melspec_db = librosa.power_to_db(mel_spectrogram, ref=1.0, amin=1e-10, top_db=top_db)

	normalized_spec = (melspec_db + top_db) / top_db
	return normalized_spec


# --- Backend Functions ---


def handle_audio_upload(audio_input):
	if audio_input is None:
		return None, 0, gr.update(visible=False), "Upload an audio file to begin.", None

	cfg = ASSETS["cfg"]
	pp_data = ASSETS["pp_data"]

	sr, audio_data = audio_input

	target_sr = pp_data["sample_rate"]
	if sr != target_sr:
		gr.Info(f"Resampling audio from {sr}Hz to {target_sr}Hz using librosa...")
		audio_data = librosa.resample(y=audio_data.astype(np.float32), orig_sr=sr, target_sr=target_sr)

	if audio_data.ndim > 1:
		audio_data = audio_data.mean(axis=1)

	# Return None for the plot output to clear it on new upload
	return (
		audio_data.astype(np.float32),
		0,
		gr.update(visible=True, value="Start Processing"),
		"Ready to process audio.",
		None,
	)


def process_single_chunk(model_type, full_audio, offset, threshold):
	if model_type == "PyTorch":
		if ASSETS["pytorch_model"] is None:
			return full_audio, offset, "PyTorch model not available. Cannot process.", None, gr.update(visible=False)
	elif model_type == "TFLite":
		if ASSETS["tflite_interpreter"] is None:
			return full_audio, offset, "TFLite model not available. Cannot process.", None, gr.update(visible=False)

	class_names = ASSETS["class_names"]
	target_width = ASSETS["target_width"]
	pp_data = ASSETS["pp_data"]

	clip_duration_s = pp_data["audio_clip_duration"]
	sample_rate = pp_data["sample_rate"]
	clip_duration_samples = int(clip_duration_s * sample_rate)

	start_sample = int(offset)
	if start_sample >= len(full_audio):
		return full_audio, offset, "Processing complete.", None, gr.update(visible=False, value="Start Processing")

	chunk = full_audio[start_sample : start_sample + clip_duration_samples]
	spectrogram = preprocess_with_librosa(chunk, pp_data)

	if target_width:
		current_width = spectrogram.shape[1]  # Spectrogram is (freq, time)
		if current_width != target_width:
			if current_width > target_width:
				spectrogram = spectrogram[:, :target_width]
			else:
				pad_width = target_width - current_width
				spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode="constant", constant_values=0)

	logits = None
	if model_type == "TFLite":
		tflite_interpreter = ASSETS["tflite_interpreter"]
		input_details = ASSETS["input_details"]
		output_details = ASSETS["output_details"]
		# NHWC format for TFLite
		spectrogram_np = spectrogram[np.newaxis, ..., np.newaxis]
		tflite_interpreter.set_tensor(input_details["index"], spectrogram_np)
		tflite_interpreter.invoke()
		logits = tflite_interpreter.get_tensor(output_details["index"])

	elif model_type == "PyTorch":
		pytorch_model = ASSETS["pytorch_model"]
		# NCHW format for PyTorch
		spectrogram_torch = torch.from_numpy(spectrogram[np.newaxis, np.newaxis, ...]).float()
		with torch.no_grad():
			logits = pytorch_model(spectrogram_torch).cpu().numpy()

	if logits is None:
		return full_audio, offset, "Error during model inference.", None, gr.update(visible=False)

	exp_logits = np.exp(logits - np.max(logits))
	probabilities = exp_logits / np.sum(exp_logits)
	probabilities = probabilities[0]

	top_prob = np.max(probabilities)
	top_idx = np.argmax(probabilities)

	start_time_s = start_sample / sample_rate
	end_time_s = min(start_sample + clip_duration_samples, len(full_audio)) / sample_rate
	time_info = f"Time: {start_time_s:.2f}s - {end_time_s:.2f}s"

	if top_prob < threshold:
		prediction_label = f"Non-match (Highest confidence: {top_prob:.2%})"
	else:
		class_label = class_names[top_idx] if class_names and top_idx < len(class_names) else f"Class Index {top_idx}"
		prediction_label = f"Prediction: {class_label} (Confidence: {top_prob:.2%})"

	output_text = f"{time_info}\n{prediction_label}"

	top_10_indices = np.argsort(probabilities)[::-1][:10]
	top_10_probs = probabilities[top_10_indices]

	plot_data = pd.DataFrame(
		{
			"Class": [
				(class_names[i] if class_names and i < len(class_names) else f"Class {i}") for i in top_10_indices
			],
			"Probability": top_10_probs.tolist(),
		}
	)

	new_offset = start_sample + clip_duration_samples
	continue_visible = new_offset < len(full_audio)
	if not continue_visible:
		output_text += "\n\nProcessing complete."

	return (
		full_audio,
		new_offset,
		output_text,
		plot_data,
		gr.update(visible=continue_visible, value="Process Next Chunk"),
	)


# --- Gradio UI ---

with gr.Blocks() as demo:
	gr.Markdown("# Audio File Validator")

	parser = argparse.ArgumentParser(
		description="Gradio App for validating audio files with TFLite and PyTorch models."
	)
	parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory.")
	args, unknown = parser.parse_known_args()

	model_choices = []
	default_model = None

	try:
		model_dir = Path(args.model_dir)
		config_path = model_dir / ".hydra" / "config.yaml"
		class_names_path = model_dir / "class_names.txt"
		preprocess_path = model_dir / "preprocess.npy"

		if not all([p.exists() for p in [config_path, class_names_path, preprocess_path]]):
			raise FileNotFoundError(
				"One or more required asset files are missing (config, class names, or preprocess data)."
			)

		cfg = OmegaConf.load(config_path)
		ASSETS["cfg"] = cfg

		with open(class_names_path, "r") as f:
			ASSETS["class_names"] = [line.strip() for line in f.readlines() if line.strip()]

		ASSETS["pp_data"] = np.load(preprocess_path, allow_pickle=True).item()

		# --- Load TFLite Model ---
		tflite_model_path = model_dir / f"{cfg.model.name}.tflite"
		if tflite_model_path.exists():
			interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
			interpreter.allocate_tensors()
			ASSETS["tflite_interpreter"] = interpreter
			ASSETS["input_details"] = interpreter.get_input_details()[0]
			ASSETS["output_details"] = interpreter.get_output_details()[0]
			ASSETS["target_width"] = ASSETS["input_details"]["shape"][2]
			model_choices.append("TFLite")
			gr.Info("TFLite model loaded successfully.")
		else:
			gr.Warning(f"TFLite model not found at '{tflite_model_path}'.")

		# --- Load PyTorch Model ---
		pytorch_model_path = model_dir / f"best_{cfg.model.name}.pth"
		if pytorch_model_path.exists():
			OmegaConf.set_struct(cfg, False)
			cfg.data.num_classes = len(ASSETS["class_names"])
			OmegaConf.set_struct(cfg, True)

			pytorch_model = Model(cfg)
			checkpoint = torch.load(pytorch_model_path, map_location="cpu")
			state_dict = checkpoint.get("model_state_dict", checkpoint)
			pytorch_model.load_state_dict(state_dict)
			pytorch_model.eval()
			ASSETS["pytorch_model"] = pytorch_model
			model_choices.append("PyTorch")
			gr.Info("PyTorch model loaded successfully.")

			if not ASSETS["target_width"] and ASSETS["tflite_interpreter"] is None:
				gr.Warning("Could not determine target input width from TFLite model. This may cause errors.")
		else:
			gr.Warning(f"PyTorch model not found at '{pytorch_model_path}'.")

		if not model_choices:
			raise RuntimeError("No TFLite or PyTorch models found in the specified directory.")

		default_model = model_choices[0]

	except Exception as e:
		ASSETS["error"] = f"Error loading assets: {e}"

	if ASSETS["error"]:
		gr.Error(ASSETS["error"])
	else:
		gr.Markdown(f"Model assets loaded from '{args.model_dir}'.")

		audio_data_state = gr.State(None)
		offset_state = gr.State(0)

		with gr.Row():
			with gr.Column(scale=1):
				gr.Markdown("### 1. Upload & Configure")
				model_selector = gr.Radio(model_choices, label="Model Type", value=default_model)
				audio_input = gr.Audio(type="numpy", label="Upload Audio File")
				threshold_slider = gr.Slider(
					minimum=0.0, maximum=1.0, value=0.01, step=0.05, label="Confidence Threshold"
				)
				process_status = gr.Textbox(label="Status", interactive=False, value="Waiting for audio file...")

			with gr.Column(scale=2):
				gr.Markdown("### 2. Analysis Results")
				with gr.Row():
					output_textbox = gr.Textbox(label="Current Chunk Prediction", lines=4, interactive=False, scale=1)
					probability_plot = gr.BarPlot(
						label="Top 10 Probabilities",
						x="Probability",
						y="Class",
						x_lim=[0, 1],
						tooltip=["Class", "Probability"],
						height=400,
					)

				process_button = gr.Button("Start Processing", visible=False)

		audio_input.upload(
			fn=handle_audio_upload,
			inputs=[audio_input],
			outputs=[audio_data_state, offset_state, process_button, process_status, probability_plot],
		)

		process_button.click(
			fn=process_single_chunk,
			inputs=[model_selector, audio_data_state, offset_state, threshold_slider],
			outputs=[audio_data_state, offset_state, output_textbox, probability_plot, process_button],
		)

if __name__ == "__main__":
	demo.launch(theme=gr.themes.Soft())
