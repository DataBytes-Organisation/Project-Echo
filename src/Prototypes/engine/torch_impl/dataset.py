import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
from pathlib import Path
import librosa
import numpy as np
import hashlib
import diskcache as dc
import soundfile as sf
import random
import math

import lmdb
import pickle

import warnings

warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*")
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")

def index_directory(directory, file_types=('.ogg', '.mp3', '.wav', '.flac')):
	audio_files = []
	labels = []
	class_names = sorted([d.name for d in Path(directory).glob('*') if d.is_dir()])
	class_to_idx = {name: i for i, name in enumerate(class_names)}

	for class_name in class_names:
		class_dir = Path(directory) / class_name
		for file_path in class_dir.glob('**/*'):
			if file_path.suffix.lower() in file_types:
				audio_files.append(str(file_path))
				labels.append(class_to_idx[class_name])

	return audio_files, labels, class_names

class SpectrogramDataset(Dataset):
	def __init__(self, audio_files, labels, cfg, audio_transforms=None, image_transforms=None, is_train=False):
		super().__init__()
		self.audio_files = audio_files
		self.labels = labels
		self.cfg = cfg
		self.audio_transforms = audio_transforms
		self.image_transforms = image_transforms
		
		self.clip_samples = int(cfg.data.audio_clip_duration * cfg.data.sample_rate)

		self.target_sample_rate = cfg.data.sample_rate
		self.target_samples = int(cfg.data.audio_clip_duration * self.target_sample_rate)
		self.is_train = is_train

		self.common_source_sr = 44100
		self.cached_resampler = torchaudio.transforms.Resample(
			orig_freq=self.common_source_sr, 
			new_freq=self.target_sample_rate
		)

		self.mel_spec = torchaudio.transforms.MelSpectrogram(
			sample_rate=cfg.data.sample_rate,
			n_fft=cfg.data.n_fft,
			hop_length=cfg.data.hop_length,
			n_mels=cfg.data.n_mels,
			f_min=cfg.data.fmin,
			f_max=cfg.data.fmax,
			power=2.0
		)
		self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=cfg.data.top_db)
		self.top_db = cfg.data.top_db

		self.use_cache = cfg.system.get('use_disk_cache', False)
		self.env = None # Lazy init for multi-processing safety
		if self.use_cache:
			self.cache_path = Path(cfg.system.cache_directory)
			self.cache_path.mkdir(parents=True, exist_ok=True)
			
			# Map size: 1TB (Virtual memory, doesn't allocate physical RAM)
			self.map_size = 1099511627776 

	def _get_cache_key(self, file_path):
		"""Generate a unique key for the cache based on file path and sample rate."""
		key_str = f"{file_path}_{self.target_sample_rate}"
		return key_str.encode('utf-8')

	def _init_db(self):
		"""Initialize LMDB environment inside the worker process."""
		self.env = lmdb.open(
			str(self.cache_path), 
			map_size=self.map_size, 
			subdir=True, 
			lock=True, # Needed for concurrent writes
			readahead=False, 
			meminit=False
		)

	def __len__(self):
		return len(self.audio_files)

	def _process_waveform_to_spec(self, waveform, file_path_for_debug="Unknown"):
		"""Helper to convert a waveform tensor (1, T) to Spectrogram (C, F, T)"""

		# Apply Audio Augmentations
		if self.audio_transforms:
			waveform_input = waveform.unsqueeze(0) # (1, 1, Time)
			try:
				augmented = self.audio_transforms(waveform_input, sample_rate=self.target_sample_rate)
				waveform = augmented.squeeze(0) # (1, Time)
			except Exception as e:
				print(f"Augmentation error on {file_path_for_debug}: {e}")

		# Convert to Spectrogram
		spec = self.mel_spec(waveform)
		spec = self.amplitude_to_db(spec)

		# Global Normalization [-80, 0] -> [0, 1]
		spec = (spec + self.top_db) / self.top_db

		# Apply Image Augmentations
		if self.image_transforms:
			if spec.dim() == 2:
				spec = spec.unsqueeze(0)
			spec = self.image_transforms(spec)

		if spec.dim() == 2:
			spec = spec.unsqueeze(0)

		return spec

	def __getitem__(self, idx):
		file_path = self.audio_files[idx]
		label = self.labels[idx]

		if self.use_cache and self.env is None:
			self._init_db()

		waveform = None

		if self.use_cache:
			key = self._get_cache_key(file_path)
			try:
				with self.env.begin(write=False) as txn:
					data = txn.get(key)
					if data:
						waveform = pickle.loads(data)
			except Exception as e:
				print(f"Cache read error for {file_path}: {e}")

		if waveform is None:
			try:
				waveform, sr = torchaudio.load(file_path)

				# Resample if necessary
				if sr != self.target_sample_rate:
					if sr == self.common_source_sr:
						waveform = self.cached_resampler(waveform)
					else:
						waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.target_sample_rate)

				# Mix to Mono (Average channels)
				if waveform.shape[0] > 1:
					waveform = torch.mean(waveform, dim=0, keepdim=True)
				
				# Write to Cache (Decoded, Resampled, Mono)
				if self.use_cache:
					try:
						with self.env.begin(write=True) as txn:
							txn.put(key, pickle.dumps(waveform))
					except Exception as e:
						print(f"Cache write error for {file_path}: {e}")

			except Exception as e:
				print(f"Error loading {file_path}: {e}")
				# Return silent tensor of correct shape on error
				return torch.zeros(self.target_samples), torch.tensor(label).long()

		# Pad or Crop to fixed length (Training Randomness happens here)
		num_samples = waveform.shape[1]

		# if num_samples > self.target_samples:
		# 	# Crop
		# 	if self.is_train:
		# 		start = random.randint(0, num_samples - self.target_samples)
		# 	else:
		# 		start = (num_samples - self.target_samples) // 2
		# 	waveform = waveform[:, start : start + self.target_samples]
		# elif num_samples < self.target_samples:
		# 	# Pad (Right padding)
		# 	padding = self.target_samples - num_samples
		# 	waveform = F.pad(waveform, (0, padding))

		if num_samples < self.target_samples:
			repeats = math.ceil(self.target_samples / num_samples)
			waveform = waveform.repeat(1, repeats)
			num_samples = waveform.shape[1]

		if self.is_train:
			if num_samples > self.target_samples:
				start = random.randint(0, num_samples - self.target_samples)
				waveform = waveform[:, start : start + self.target_samples]
			
			spec = self._process_waveform_to_spec(waveform, file_path)
			return spec, torch.tensor(label).long()

		else:
			chunks = []
			
			# Case A: Exact match (after duplication logic)
			if num_samples == self.target_samples:
				chunks.append(waveform)

			# Case B: Break into fixed chunks
			else:
				# Stride = target_samples (No overlap, just tiling)
				for start in range(0, num_samples, self.target_samples):
					end = start + self.target_samples
					if end <= num_samples:
						chunks.append(waveform[:, start:end])
					else:
						chunks.append(waveform[:, -self.target_samples:])

			# Process all chunks into spectrograms
			specs = []
			for chunk_wave in chunks:
				# chunk_wave is (1, Time)
				s = self._process_waveform_to_spec(chunk_wave, file_path)
				specs.append(s)

			# Stack them: (Num_Chunks, C, F, T)
			stacked_specs = torch.stack(specs)
			
			# Duplicate labels: (Num_Chunks)
			stacked_labels = torch.tensor([label] * len(specs)).long()

			return stacked_specs, stacked_labels
