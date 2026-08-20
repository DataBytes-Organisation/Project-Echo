"""
The SpecAugment implementation, with the modifications from
"Review of current SpecAugment implementation and its connection to the
training dataloader.md" applied. Inline comments reference the exact Chunk /
Modification labels used in that document.

Started as a testing copy under a local torch_impl/ folder, then
moved into its permanent home here at src/prototypes/engine/augmentation/
once the changes were verified.
"""

import torch
import torch.nn as nn
import random

# --- Chunk 1 modification ------------
# Original imported `Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift`
# from `audiomentations`, and `transforms` from `torchvision`. Removed here:
# none of them were used anywhere in the original file (Chunk 1 finding).


class SpecAugment(nn.Module):
	"""
	Spectrogram augmentation module.
	Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
	(https://arxiv.org/abs/1904.08779)
	"""
	# Chunk 2: no modification needed - class declaration and docstring unchanged.

	def __init__(
		self,
		p=0.5,
		n_freq_mask=2,
		n_time_mask=2,
		# --- Chunk 3, Modification 1 --------
		# "these five values must be configured in the
		# project's configuration files (used by the
		# project's Hydra config system)." These two stay as optional
		# fixed-pixel overrides (default None) - the values actually used at
		# training time come from config/augmentation/*.yaml once wired up
		# through Hydra, not from the defaults written here.
		freq_mask_param=None,
		time_mask_param=None,
		# --- Chunk 3, Modification 3 ------------
		# "add ratio-based options like freq_mask_ratio and time_mask_ratio
		# (percentages, not fixed pixel counts)... a percentage-based setting
		# can be adaptive and is therefore a better choice." These are now the
		# new defaults; the fixed-pixel arguments above still work if
		# explicitly supplied.
		freq_mask_ratio=0.15,
		time_mask_ratio=0.10,
		# --- Chunk 6, Modification 1 --------------
		# "we must add a hard cap on the total combined coverage (e.g. never
		# more than 40%) to strictly prevent the covering-most-timeframes issue from
		# happening no matter how the audio clip length or audio settings
		# change later."
		max_total_time_ratio=0.4,
		# --- Chunk 3, Modification 2 -----------------------------------------------
		# "add mask_value (so the covering colour is a choice, not fixed)."
		# "zero" reproduces the exact original behaviour.
		mask_value="zero",
	):
		super().__init__()
		self.p = p
		self.n_freq_mask = n_freq_mask
		self.n_time_mask = n_time_mask
		self.freq_mask_param = freq_mask_param
		self.time_mask_param = time_mask_param
		self.freq_mask_ratio = freq_mask_ratio
		self.time_mask_ratio = time_mask_ratio
		self.max_total_time_ratio = max_total_time_ratio
		self.mask_value = mask_value

	def _fill_value(self, x):
		# Supports Chunk 3, Modification 2 (the mask_value option).
		if self.mask_value == "zero":
			return 0.0
		elif self.mask_value == "min":
			return x.min()
		elif self.mask_value == "mean":
			return x.mean()
		else:
			raise ValueError(f"Unknown mask_value: {self.mask_value!r}")

	def forward(self, x):
		# Chunk 4: no modification needed to this part of the logic.
		if random.random() > self.p:
			return x

		C, F, T = x.shape
		# set the fill value as per the chosen mask_value option (0, min, mean)
		fill = self._fill_value(x)

		# -----Chunk 5 (original lines 30-36): frequency masking---------
		freq_param = (
			self.freq_mask_param
			if self.freq_mask_param is not None
			else max(1, int(F * self.freq_mask_ratio))  # Chunk 3, Modification 3: number of blacked out freqs per freq-strip - a percentage (freq_mask_ratio) of total freqs (min 1 pixel) - this is the default case because freq_mask_param=None by default. The other case is using freq_mask_param if a value is explicitly set.
		)
		for _ in range(self.n_freq_mask):
			# updated to use the new freq_param (the old file uses freq_mask_param)
			f_param = min(freq_param, F)

			# --- Modification needed at line 34 (Chunk 5) ---------------
			# Original: f = random.randint(0, F_param)
			# This could return 0, producing a mask with zero width that
			# covers nothing - "a configured mask should always actually mask
			# something." Lower bound changed from 0 to 1.
			# ------------------------------------------------------------------------
			f = random.randint(1, max(1, f_param)) # use max to set the lower bound to 1 to prevent configuring e.g. freq_mask_param=0.8

			f0 = random.randint(0, F - f)
			x[:, f0 : f0 + f, :] = fill  # fill is now configurable (was hardcoded 0)

		# ------Chunk 6 (original lines 38-43): time masking-------------
		time_param = (
			self.time_mask_param
			if self.time_mask_param is not None
			else max(1, int(T * self.time_mask_ratio))  # Chunk 3, Modification 3 (same logic as freq_param but for the time-strip)
		)

		# --- Chunk 6, Modification 1: hard cap on total combined coverage -----
		# Without this cap, the original defaults (time_mask_param=80,
		# n_time_mask=2) could black out up to ~80% of a 201-column clip (the
		# headline finding in the review). This ensures no combination of
		# settings can ever exceed max_total_time_ratio of the clip, even if
		# old fixed-pixel values are passed in directly.
		max_total_time = max(1, int(T * self.max_total_time_ratio)) # max total timeframes to black out
		total_masked = 0 # total blacked out timeframes so far, initially 0

		for _ in range(self.n_time_mask):
			# updated from using time_mask_param to using the new time_param
			t_param = min(time_param, T)
			# remaining allowed timeframes to black out after blacking out some in previous iterations (total_masked so far)
			remaining = max_total_time - total_masked
			if remaining <= 0:
				break  # cap already reached - stop adding more masks

			# --- Chunk 6, Modification 2: same zero-width fix as Chunk 5 ----
			# Original: t = random.randint(0, T_param)
			t = random.randint(1, max(1, min(t_param, remaining))) # choose remaining allowed timeframes to black out if it is < than the original blacked out timeframes per strip

			t0 = random.randint(0, T - t)
			x[:, :, t0 : t0 + t] = fill # configurable instead of hardcoded
			total_masked += t # accumulate to total blacked out timeframes to determine remaining ones in next iterations

		return x

	def __repr__(self):
		# Not one of the listed modifications, but needed so a training log
		# can show which augmentation settings were actually active for a run.
		return (
			f"SpecAugment(p={self.p}, n_freq_mask={self.n_freq_mask}, n_time_mask={self.n_time_mask}, "
			f"freq_mask_param={self.freq_mask_param}, time_mask_param={self.time_mask_param}, "
			f"freq_mask_ratio={self.freq_mask_ratio}, time_mask_ratio={self.time_mask_ratio}, "
			f"max_total_time_ratio={self.max_total_time_ratio}, mask_value={self.mask_value})"
		)

	# Chunk 7 (original line 45, `return x`): folded into forward() above -
	# no separate modification needed here.

# --- Duplicate implementation note --------------------------------------------
# A second written copy of this class used to exist in
# _single.py. In the review doc: "This needs to be removed and replaced with
# an import of augment.py after implementing the above modifications." That
# fix was made while this file still lived in a local torch_impl/ testing
# copy - _single.py itself wasn't carried into this repo location, since
# nothing here depends on it