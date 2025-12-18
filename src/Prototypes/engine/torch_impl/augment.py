import torch
import torch.nn as nn
import random

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from torchvision import transforms

class SpecAugment(nn.Module):
	"""
	Spectrogram augmentation module.
	Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
	(https://arxiv.org/abs/1904.08779)
	"""
	def __init__(self, p=0.5, freq_mask_param=30, time_mask_param=80, n_freq_mask=2, n_time_mask=2):
		super().__init__()
		self.p = p
		self.freq_mask_param = freq_mask_param
		self.time_mask_param = time_mask_param
		self.n_freq_mask = n_freq_mask
		self.n_time_mask = n_time_mask

	def forward(self, x):
		if random.random() > self.p:
			return x

		C, F, T = x.shape
		
		# x is (C, F, T)
		for _ in range(self.n_freq_mask):
			F_param = min(self.freq_mask_param, F)

			f = random.randint(0, F_param)
			f0 = random.randint(0, F - f)
			x[:, f0:f0+f, :] = 0
		
		for _ in range(self.n_time_mask):
			T_param = min(self.time_mask_param, T)

			t = random.randint(0, T_param)
			t0 = random.randint(0, T - t)
			x[:, :, t0:t0+t] = 0
			
		return x
