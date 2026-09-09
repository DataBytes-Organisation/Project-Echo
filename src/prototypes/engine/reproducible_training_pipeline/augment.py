import random

import torch.nn as nn


class SpecAugment(nn.Module):
	"""
	Spectrogram augmentation module.
	Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
	(https://arxiv.org/abs/1904.08779)

	Supports two ways of sizing each masked strip, selected per axis by
	which pair of constructor args is supplied:
	  - Legacy pixel API: freq_mask_param / time_mask_param (fixed max width
	    in bins/timeframes). Used by the `original_unfixed_reference` preset
	    to reproduce the original, uncapped masking behaviour.
	  - Ratio API: freq_mask_ratio / time_mask_ratio (max width as a
	    fraction of the spectrogram's total size on that axis). Used by the
	    default/light/heavy presets so mask size scales with clip length
	    instead of being a fixed pixel count.
	Exactly one of each pair must be given; mixing pixel-freq with
	ratio-time (or vice versa) is allowed but unused by any current preset.
	"""

	def __init__(
		self,
		p=0.5,
		n_freq_mask=2,
		n_time_mask=2,
		freq_mask_param=None,
		time_mask_param=None,
		freq_mask_ratio=None,
		time_mask_ratio=None,
		max_total_time_ratio=1.0,
		mask_value="zero",
	):
		super().__init__()

		if (freq_mask_param is None) == (freq_mask_ratio is None):
			raise ValueError(
				"SpecAugment: specify exactly one of freq_mask_param or freq_mask_ratio, not both/neither."
			)
		if (time_mask_param is None) == (time_mask_ratio is None):
			raise ValueError(
				"SpecAugment: specify exactly one of time_mask_param or time_mask_ratio, not both/neither."
			)
		if freq_mask_ratio is not None and not (0.0 <= freq_mask_ratio <= 1.0):
			raise ValueError(f"SpecAugment: freq_mask_ratio must be in [0.0, 1.0], got {freq_mask_ratio}.")
		if time_mask_ratio is not None and not (0.0 <= time_mask_ratio <= 1.0):
			raise ValueError(f"SpecAugment: time_mask_ratio must be in [0.0, 1.0], got {time_mask_ratio}.")
		if not (0.0 <= max_total_time_ratio <= 1.0):
			raise ValueError(f"SpecAugment: max_total_time_ratio must be in [0.0, 1.0], got {max_total_time_ratio}.")
		if mask_value != "zero":
			raise ValueError(f"SpecAugment: unsupported mask_value '{mask_value}'; only 'zero' is currently implemented.")

		self.p = p
		self.n_freq_mask = n_freq_mask
		self.n_time_mask = n_time_mask
		self.freq_mask_param = freq_mask_param
		self.time_mask_param = time_mask_param
		self.freq_mask_ratio = freq_mask_ratio
		self.time_mask_ratio = time_mask_ratio
		self.max_total_time_ratio = max_total_time_ratio
		self.mask_value = mask_value

	def _fill(self, x, index_slice):
		# Only "zero" is implemented today; validated in __init__.
		x[index_slice] = 0.0

	def _strip_width(self, param, ratio, axis_size):
		"""Draw a random strip width, capped either by a fixed param (pixel API) or a fraction of axis_size (ratio API)."""
		cap = param if param is not None else int(axis_size * ratio)
		cap = min(cap, axis_size)
		return random.randint(0, cap)

	def forward(self, x):
		if random.random() > self.p:
			return x

		C, F, T = x.shape

		# Frequency strips: no cumulative cap, matches every current preset.
		for _ in range(self.n_freq_mask):
			width = self._strip_width(self.freq_mask_param, self.freq_mask_ratio, F)
			if width <= 0:
				continue
			f0 = random.randint(0, F - width)
			self._fill(x, (slice(None), slice(f0, f0 + width), slice(None)))

		# Time strips: cumulative width is capped at max_total_time_ratio * T.
		# With max_total_time_ratio=1.0 the cap can never bind (equals the
		# full clip width), reproducing the "no safety cap" behaviour used
		# by original_unfixed_reference.yaml through this same code path.
		max_total_time_frames = int(T * self.max_total_time_ratio)
		total_time_masked = 0
		for _ in range(self.n_time_mask):
			width = self._strip_width(self.time_mask_param, self.time_mask_ratio, T)
			width = min(width, max_total_time_frames - total_time_masked)
			if width <= 0:
				continue
			t0 = random.randint(0, T - width)
			self._fill(x, (slice(None), slice(None), slice(t0, t0 + width)))
			total_time_masked += width

		return x
