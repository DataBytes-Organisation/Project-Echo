import argparse
import torch
import torchaudio
import librosa
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import sys
from scipy import signal

# This class is a direct copy from deploy_model.py for a self-contained test
class TorchAudioPreprocessor(torch.nn.Module):
	"""
	A TorchScript-compatible preprocessor for audio data using torchaudio.
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
			center=False,  # We will pad manually
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
			return torch.nn.functional.pad(waveform, (0, pad_amount))
		return waveform

	def forward(self, waveform: torch.Tensor) -> torch.Tensor:
		if waveform.dim() == 1:
			waveform = waveform.unsqueeze(0)

		waveform = self._pad_or_truncate(waveform)

		# Manually pad for STFT consistency
		padding = self.melspectrogram.n_fft // 2
		waveform = torch.nn.functional.pad(waveform, (padding, padding), mode='reflect')

		melspec = self.melspectrogram(waveform)
		melspec_db = self.amplitude_to_db(melspec)
		normalized_spec = (melspec_db + self.top_db) / self.top_db
		return normalized_spec

def preprocess_with_librosa(waveform: np.ndarray, cfg: OmegaConf, mel_filters: np.ndarray) -> np.ndarray:
	"""
	An equivalent preprocessor for audio data using librosa and a pre-computed
	torchaudio filter bank.
	"""
	data_cfg = cfg.data
	sample_rate = data_cfg.sample_rate
	target_duration_samples = int(data_cfg.audio_clip_duration * sample_rate)
	n_fft = data_cfg.n_fft
	hop_length = data_cfg.hop_length

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
	# The torchaudio filter bank has shape (n_freqs, n_mels).
	# Perform (S.T @ FB).T to match torchaudio's matmul logic.
	mel_spectrogram = (stft_power.T @ mel_filters).T

	# Convert to dB scale
	top_db = float(data_cfg.top_db)
	melspec_db = librosa.power_to_db(
		mel_spectrogram,
		ref=1.0,
		amin=1e-10,
		top_db=top_db
	)

	# Normalize
	normalized_spec = (melspec_db + top_db) / top_db

	# Add a channel dimension to match torchaudio's (C, H, W) output
	return normalized_spec[np.newaxis, :, :]


def main():
	parser = argparse.ArgumentParser(
		description="Compare torchaudio and librosa audio preprocessing."
	)
	parser.add_argument(
		"--config_path",
		type=str,
		required=True,
		help="Path to the .hydra/config.yaml file from a trained model."
	)
	parser.add_argument(
		"--audio_file",
		type=str,
		required=True,
		help="Path to an audio file to test preprocessing on."
	)
	args = parser.parse_args()

	config_path = Path(args.config_path)
	audio_file = Path(args.audio_file)

	if not config_path.exists():
		print(f"Error: Config file not found at '{config_path}'")
		sys.exit(1)
	if not audio_file.exists():
		print(f"Error: Audio file not found at '{audio_file}'")
		sys.exit(1)

	print(f"Loading configuration from '{config_path}'")
	cfg = OmegaConf.load(config_path)

	print(f"Loading audio from '{audio_file}'")
	waveform_torch, sr = torchaudio.load(audio_file)

	# Ensure single channel for simplicity
	if waveform_torch.shape[0] > 1:
		waveform_torch = torch.mean(waveform_torch, dim=0, keepdim=True)

	print(f"Audio loaded. Original SR: {sr}Hz. Shape: {waveform_torch.shape}")

	# Resample if necessary
	target_sr = cfg.data.sample_rate
	if sr != target_sr:
		print(f"Resampling audio from {sr}Hz to {target_sr}Hz...")
		resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
		waveform_torch = resampler(waveform_torch)

	# Use the exact same waveform for both pipelines to isolate the test
	waveform_numpy = waveform_torch.squeeze(0).numpy()

	# Generate reference filter bank from config
	print("\n--- Generating reference torchaudio filter bank ---")
	mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
		sample_rate=cfg.data.sample_rate,
		n_fft=cfg.data.n_fft,
		n_mels=cfg.data.n_mels,
		f_min=cfg.data.fmin,
		f_max=cfg.data.fmax,
		norm=None,
	)
	# Shape is (n_freqs, n_mels), which is what our manual matmul expects
	torchaudio_fb = mel_spectrogram_transform.mel_scale.fb.numpy()
	print(f"  Filter bank generated with shape: {torchaudio_fb.shape}")

	print("\n--- Running torchaudio Mel preprocessing ---")
	torchaudio_preprocessor = TorchAudioPreprocessor(cfg)
	torchaudio_spec = torchaudio_preprocessor(waveform_torch)
	print(f"  Output shape: {torchaudio_spec.shape}")

	print("\n--- Running librosa Mel preprocessing ---")
	librosa_spec = preprocess_with_librosa(waveform_numpy, cfg, torchaudio_fb)
	print(f"  Output shape: {librosa_spec.shape}")

	print("\n--- Comparing final Mel spectrogram outputs ---")

	# Convert torchaudio tensor to numpy for comparison
	torchaudio_spec_np = torchaudio_spec.numpy()

	if torchaudio_spec_np.shape != librosa_spec.shape:
		print("Error: Output shapes do not match!")
		print(f"  Torchaudio: {torchaudio_spec_np.shape}")
		print(f"  Librosa:	{librosa_spec.shape}")
		sys.exit(1)

	absolute_difference = np.abs(torchaudio_spec_np - librosa_spec)
	max_diff = np.max(absolute_difference)
	mean_diff = np.mean(absolute_difference)

	print(f"  Max absolute difference:  {max_diff:.8f}")
	print(f"  Mean absolute difference: {mean_diff:.8f}")

	tolerance = 1e-5
	if max_diff < tolerance:
		print(f"\nPASSED: The difference is within the tolerance of {tolerance}.")
	else:
		print(f"\nWARNING: The difference exceeds the tolerance of {tolerance}.")
		print("  This may be acceptable, but review the differences if results are poor.")

if __name__ == "__main__":
	main()
