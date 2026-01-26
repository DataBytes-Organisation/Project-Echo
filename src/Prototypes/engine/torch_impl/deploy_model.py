import argparse
from pathlib import Path
import sys
import os
import subprocess
from typing import Tuple

# For TFLite conversion - import before torch to avoid conflicts
import onnx
import tensorflow as tf
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import OmegaConf

from model import Model

import onnx.numpy_helper
import json

import onnxsim
import shutil

class TorchscriptModelWrapper(nn.Module):
	"""
	A TorchScript-compatible wrapper for the classifier model that includes
	thresholding logic directly in its forward pass.
	"""
	def __init__(self, model: nn.Module):
		super().__init__()
		self.model = model

	def forward(self, input_tensor: torch.Tensor, threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
		# Unconditionally add a batch dimension
		logits = self.model(input_tensor.unsqueeze(0))

		probs = F.softmax(logits, dim=1)
		probs_flat = probs.squeeze(0)

		top_prob, top_idx = torch.max(probs_flat, dim=0)
		unknown_idx = torch.tensor(-1, dtype=top_idx.dtype, device=top_idx.device)

		# Use torch.where for conditional selection, which is TorchScript-friendly
		result_idx = torch.where(top_prob >= threshold, top_idx, unknown_idx)

		return result_idx, probs


class Preprocessor(nn.Module):
	"""
	A TorchScript-compatible preprocessor for audio data.
	"""
	def __init__(self, cfg: OmegaConf):
		super().__init__()
		data_cfg = cfg.data
		self.target_duration_samples = int(data_cfg.audio_clip_duration * data_cfg.sample_rate)

		self.melspectrogram = torchaudio.transforms.MelSpectrogram(
			sample_rate=data_cfg.sample_rate,
			n_fft=data_cfg.n_fft,
			hop_length=data_cfg.hop_length,
			n_mels=data_cfg.n_mels,
			f_min=data_cfg.fmin,
			f_max=data_cfg.fmax,
			power=2.0,
		)

		self.top_db = float(data_cfg.top_db)
		self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
			stype='power', top_db=self.top_db
		)

	def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
		current_len = waveform.shape[-1]

		if current_len > self.target_duration_samples:
			return waveform[..., :self.target_duration_samples]
		elif current_len < self.target_duration_samples:
			pad_amount = self.target_duration_samples - current_len
			return F.pad(waveform, (0, pad_amount))

		return waveform

	def forward(self, waveform: torch.Tensor) -> torch.Tensor:
		if waveform.dim() == 1:
			waveform = waveform.unsqueeze(0)

		waveform = self._pad_or_truncate(waveform)
		melspec = self.melspectrogram(waveform)
		melspec_db = self.amplitude_to_db(melspec)
		normalized_spec = (melspec_db + self.top_db) / self.top_db

		return normalized_spec


def simplify_onnx_model(input_path: Path, output_path: Path):
	"""
	Runs onnx-simplifier on the model using a subprocess.
	This prevents the main script from crashing if onnxsim encounters a
	Segmentation Fault (which can happen with version mismatches).
	"""

	print(f"  Attempting to simplify model: {input_path}")

	try:
		# Run via python -m onnxsim to isolate the process
		cmd = [sys.executable, "-m", "onnxsim", str(input_path), str(output_path)]

		result = subprocess.run(cmd, capture_output=True, text=True)

		if result.returncode != 0:
			print(f"  Warning: onnxsim failed/crashed (Return Code: {result.returncode}).")
			if result.returncode == -11 or result.returncode == 139:
				 print("  Error type: Segmentation Fault (Core Dumped) in onnxsim.")

			print(f"  onnxsim stderr: {result.stderr}")
			print("  Skipping simplification and using original ONNX model.")
			shutil.copy(str(input_path), str(output_path))
		else:
			print(f"  Simplified model saved successfully to: {output_path}")

	except Exception as e:
		print(f"  Error executing onnxsim subprocess: {e}")
		print("  Skipping simplification.")
		shutil.copy(str(input_path), str(output_path))


def fix_onnx_model(input_path: Path, output_path: Path):
	"""
	Scans the ONNX graph for operators that strictly require INT32 inputs for TFLite
	(Expand, Reshape, Tile, Reduce ops) and converts their inputs from INT64 to INT32.
	"""
	print(f"  Scanning for INT64 inputs to convert to INT32 in {input_path}...")
	model = onnx.load(str(input_path))
	graph = model.graph

	# Define sensitive inputs: {OpType: [index_of_input_that_must_be_int32]}
	target_ops = {
		'Expand': [1],
		'Reshape': [1],
		'Tile': [1],
		'ReduceL2': [1],
		'ReduceMean': [1],
		'ReduceSum': [1],
		'ReduceMax': [1],
		'ReduceMin': [1],
		'OneHot': [1],
		'Gather': [1],
		'Slice': [1, 2, 3, 4]
	}

	# Find all tensor names connected to these sensitive inputs
	sensitive_tensor_names = set()
	for node in graph.node:
		if node.op_type in target_ops:
			indices = target_ops[node.op_type]
			for idx in indices:
				if len(node.input) > idx:
					sensitive_tensor_names.add(node.input[idx])

	made_changes = False

	# Fix Initializers (Global Weights)
	new_initializers = []
	for initializer in graph.initializer:
		if initializer.name in sensitive_tensor_names and initializer.data_type == onnx.TensorProto.INT64:
			made_changes = True
			# Convert data to int32
			data = onnx.numpy_helper.to_array(initializer)
			new_data = data.astype(np.int32)
			new_initializer = onnx.helper.make_tensor(
				name=initializer.name,
				data_type=onnx.TensorProto.INT32,
				dims=new_data.shape,
				vals=new_data.flatten()
			)
			new_initializers.append(new_initializer)
		else:
			new_initializers.append(initializer)

	# Fix Constant Nodes (Embedded Constants in the graph flow)
	new_nodes = []
	for node in graph.node:
		if node.op_type == 'Constant' and node.output[0] in sensitive_tensor_names:
			attr = next((a for a in node.attribute if a.name == 'value'), None)
			if attr and attr.t.data_type == onnx.TensorProto.INT64:
				made_changes = True
				tensor = attr.t
				val_array = onnx.numpy_helper.to_array(tensor)
				new_data = val_array.astype(np.int32)
				new_tensor = onnx.helper.make_tensor(
					name=f"{node.name}_fixed",
					data_type=onnx.TensorProto.INT32,
					dims=new_data.shape,
					vals=new_data.flatten()
				)
				new_node = onnx.helper.make_node(
					'Constant',
					inputs=[],
					outputs=node.output,
					name=node.name,
					value=new_tensor
				)
				new_nodes.append(new_node)
				continue

		new_nodes.append(node)

	# Save the fixed model
	if made_changes:
		new_graph = onnx.helper.make_graph(
			new_nodes, graph.name, graph.input, graph.output, new_initializers
		)
		new_model = onnx.helper.make_model(new_graph, opset_imports=model.opset_import)
		onnx.save(new_model, str(output_path))
		print(f"  Fixed model saved to: {output_path}")
	else:
		print("  No problematic int64 tensors found. Copying input to output.")
		import shutil
		shutil.copy(str(input_path), str(output_path))


def main(args):
	"""Main function to perform the conversion and packaging."""
	model_dir = Path(args.model_dir)

	if not model_dir.is_dir():
		print(f"Error: '{model_dir}' is not a valid directory.")
		sys.exit(1)

	config_path = model_dir / ".hydra" / "config.yaml"
	if not config_path.exists():
		print(f"Error: Could not find configuration file at '{config_path}'.")
		sys.exit(1)

	print(f"Loading configuration from {config_path}")
	cfg = OmegaConf.load(config_path)

	class_names_path = model_dir / "class_names.txt"
	if not class_names_path.exists():
		print(f"Error: 'class_names.txt' not found in '{model_dir}'.")
		sys.exit(1)

	with open(class_names_path, "r") as f:
		num_classes = len([line for line in f if line.strip()])

	print(f"Determined model has {num_classes} classes.")

	OmegaConf.set_struct(cfg, False)
	cfg.data.num_classes = num_classes
	OmegaConf.set_struct(cfg, True)

	# Load PyTorch Model
	print("\n--- Step 1: Loading PyTorch model ---")
	try:
		model = Model(cfg)
		model_path = model_dir / f"best_{cfg.model.name}.pth"
		if not model_path.exists():
			print(f"Error: Could not find model checkpoint '{model_path}'.")
			sys.exit(1)

		print(f"  Loading model weights from '{model_path}'")
		checkpoint = torch.load(model_path, map_location='cpu')

		if 'model_state_dict' in checkpoint:
			model.load_state_dict(checkpoint['model_state_dict'])
		else:
			model.load_state_dict(checkpoint)

		if cfg.run.get('quantise', False):
			print("  Model was trained with QAT, converting to final quantized model...")
			model.quantise()

		model.eval()
		print("  Model loaded successfully.")
	except Exception as e:
		print(f"  Error loading model: {e}")
		sys.exit(1)

	print("\n--- Step 2: Creating and scripting the classifier model wrapper ---")
	try:
		wrapper = TorchscriptModelWrapper(model)
		wrapper.eval()
		scripted_wrapper = torch.jit.script(wrapper)
		classifier_path = model_dir / "model.pt"
		scripted_wrapper.save(classifier_path)
		print(f"  TorchScript wrapper model saved successfully to '{classifier_path}'")
	except Exception as e:
		print(f"  Error scripting classifier: {e}")
		sys.exit(1)

	if args.torchscript_only:
		print("\n--- Deployment artifact generation complete (TorchScript only). ---")
		sys.exit(0)

	print("\n--- Step 3: Exporting model to ONNX ---")
	onnx_path = model_dir / f"{cfg.model.name}.onnx"
	try:
		# Dummy input shape (Batch, Channels, Mels, Time)
		dummy_input_shape = (1, 1, cfg.data.n_mels, 128)
		dummy_input = torch.randn(dummy_input_shape, requires_grad=False)

		print(f"  Exporting to ONNX format at '{onnx_path}'...")
		torch.onnx.export(
			model,
			dummy_input,
			str(onnx_path),
			export_params=True,
			opset_version=18,
			do_constant_folding=True,
			input_names=['input'],
			output_names=['output'],
			dynamo=False,
		)
		print("  ONNX model saved successfully.")
	except Exception as e:
		print(f"  Error exporting to ONNX: {e}")
		sys.exit(1)

	# --- Simplify ONNX ---
	# This is crucial for fixing 'Expand' and constant folding issues
	print("\n--- Step 3.5: Simplifying ONNX model (Folding constants) ---")
	simplified_onnx_path = onnx_path.with_suffix('.simplified.onnx')
	try:
		simplify_onnx_model(onnx_path, simplified_onnx_path)
	except Exception as e:
		print(f"  Error during ONNX simplification: {e}")
		sys.exit(1)

	# Fix INT64 inputs on the SIMPLIFIED model ---
	print("\n--- Step 3.6: Fixing INT64 inputs for TFLite compatibility ---")
	fixed_onnx_path = onnx_path.with_suffix('.fixed.onnx')
	try:
		fix_onnx_model(simplified_onnx_path, fixed_onnx_path)
	except Exception as e:
		print(f"  Error fixing ONNX model types: {e}")
		sys.exit(1)

	# Convert ONNX to TFLite
	print("\n--- Step 4: Converting ONNX to TFLite ---")
	tflite_path = model_dir / f"{cfg.model.name}.tflite"
	tf_saved_model_path = model_dir / f"{cfg.model.name}_savedmodel"

	try:
		print(f"  [1/3] Converting ONNX to TensorFlow SavedModel using onnx2tf...")

		command = [
			"onnx2tf",
			"-i", str(fixed_onnx_path),
			"-o", str(tf_saved_model_path),
			"--non_verbose",
			"-eatfp16",  # Enable experimental FP16 support
			"-b", "1",   # Force batch size 1
		]

		subprocess.run(command, check=True, capture_output=True)
		print("  TensorFlow SavedModel generated successfully.")

		print(f"  [2/3] Converting TensorFlow SavedModel to TFLite at '{tflite_path}'...")
		converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_path))
		converter.optimizations = [tf.lite.Optimize.DEFAULT]
		converter.target_spec.supported_ops = [
			tf.lite.OpsSet.TFLITE_BUILTINS,
			tf.lite.OpsSet.SELECT_TF_OPS
		]
		tflite_model = converter.convert()

		with open(tflite_path, 'wb') as f:
			f.write(tflite_model)
		print("  [3/3] TFLite model saved successfully.")

		# Cleanup
		import shutil
		if tf_saved_model_path.exists():
			print(f"  Cleaning up temporary directory '{tf_saved_model_path}'...")
			shutil.rmtree(str(tf_saved_model_path))

	except Exception as e:
		print(f"\n  Error during TFLite conversion: {e}")
		if isinstance(e, subprocess.CalledProcessError):
			print(f"  onnx2tf stdout: {e.stdout.decode()}")
			print(f"  onnx2tf stderr: {e.stderr.decode()}")
		sys.exit(1)

	print("\n--- Step 5: Generating and saving preprocessing data ---")
	preprocess_path = model_dir / "preprocess.npy"
	try:
		mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
			sample_rate=cfg.data.sample_rate,
			n_fft=cfg.data.n_fft,
			n_mels=cfg.data.n_mels,
			f_min=cfg.data.fmin,
			f_max=cfg.data.fmax,
			norm=None,
		)
		filter_bank = mel_spectrogram_transform.mel_scale.fb.numpy()

		preprocess_data = {
			'sample_rate': cfg.data.sample_rate,
			'n_fft': cfg.data.n_fft,
			'hop_length': cfg.data.hop_length,
			'top_db': cfg.data.top_db,
			'audio_clip_duration': cfg.data.audio_clip_duration,
			'mel_filters': filter_bank,
		}

		np.save(preprocess_path, preprocess_data)
		print("  Preprocessing data saved successfully.")

	except Exception as e:
		print(f"\n  Error generating preprocessing data: {e}")
		sys.exit(1)


	print("\n" + "=" * 60)
	print("Generated Artifacts Summary")
	print("=" * 60)
	onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
	tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)
	print(f"  ONNX model size:	{onnx_size:.2f} MB")
	print(f"  TFLite model size:  {tflite_size:.2f} MB")
	print("=" * 60)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Generate all deployment artifacts (TorchScript, ONNX, TFLite).",
		formatter_class=argparse.RawTextHelpFormatter
	)
	parser.add_argument(
		"model_dir",
		type=str,
		help="Path to the directory of the trained model."
	)
	parser.add_argument(
		'--torchscript-only',
		action='store_true',
		help='Only generate the TorchScript model and exit.'
	)
	args = parser.parse_args()
	main(args)
