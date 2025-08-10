import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
import librosa
import numpy as np
import hashlib
import diskcache as dc
import soundfile as sf

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

"""
TODO: 
- [  ] use lmdb for cache to improve performance
"""
class SpectrogramDataset(Dataset):
	def __init__(self, audio_files, labels, cfg, audio_transforms=None, image_transforms=None):
		super().__init__()
		self.audio_files = audio_files
		self.labels = labels
		self.cfg = cfg
		self.audio_transforms = audio_transforms
		self.image_transforms = image_transforms
		
		self.clip_samples = int(cfg.data.audio_clip_duration * cfg.data.sample_rate)

		if cfg.system.use_disk_cache:
			self.cache = dc.Cache(cfg.system.cache_directory, size_limit=10**10)
		else:
			self.cache = None

	def _get_cache_key(self, file_path):
		m = hashlib.sha256()

		m.update(str(file_path).encode())
		m.update(str(self.cfg.data).encode())

		if self.audio_transforms:
			m.update(str(self.cfg.augmentations.audio).encode())

		if self.image_transforms:
			m.update(str(self.cfg.augmentations.image).encode())

		return m.hexdigest()

	def __len__(self):
		return len(self.audio_files)

	def __getitem__(self, idx):
		file_path = self.audio_files[idx]
		label = self.labels[idx]

		cache_key = None
		if self.cache:
			cache_key = self._get_cache_key(file_path)
			if cache_key in self.cache:
				spectrogram = self.cache[cache_key]
				return torch.from_numpy(spectrogram).float(), torch.tensor(label).long()

		try:
			audio_data, sr = librosa.load(file_path, sr=self.cfg.data.sample_rate, mono=True)
		except Exception as e:
			print(f"Error loading {file_path}: {e}")
			audio_data = np.zeros(self.clip_samples, dtype=np.float32)

		# Randomly sample a clip during training
		if len(audio_data) > self.clip_samples:
			if self.is_train:
				start = np.random.randint(0, len(audio_data) - self.clip_samples)
			else:
				start = (len(audio_data) - self.clip_samples) // 2
			audio_data = audio_data[start : start + self.clip_samples]
		elif len(audio_data) < self.clip_samples:
			padding = self.clip_samples - len(audio_data)
			audio_data = np.pad(audio_data, (0, padding), 'constant')

		if self.audio_transforms:
			audio_data = self.audio_transforms(samples=audio_data, sample_rate=self.cfg.data.sample_rate)

		mel_spec = librosa.feature.melspectrogram(
			y=audio_data,
			sr=self.cfg.data.sample_rate,
			n_fft=self.cfg.data.n_fft,
			hop_length=self.cfg.data.hop_length,
			n_mels=self.cfg.data.n_mels,
			fmin=self.cfg.data.fmin,
			fmax=self.cfg.data.fmax
		)
		
		mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.cfg.data.top_db)
		
		mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)

		spectrogram = np.expand_dims(mel_spec_db, axis=0)
		
		if self.image_transforms:
			spectrogram_tensor = torch.from_numpy(spectrogram).float()
			spectrogram_tensor = self.image_transforms(spectrogram_tensor)
			spectrogram = spectrogram_tensor.numpy()

		if self.cache:
			self.cache[cache_key] = spectrogram

		return torch.from_numpy(spectrogram).float(), torch.tensor(label).long()
