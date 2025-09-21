import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import CosineLinear

import math

class ECA(nn.Module):
	def __init__(self, in_channels, gamma=2, b=1):
		super().__init__()

		t = int(abs((math.log(in_channels, 2) + b) / gamma))
		k_size = t if t % 2 else t + 1

		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		# Squeeze
		y = self.avg_pool(x)

		# Excite
		y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
		y = self.sigmoid(y)

		return x * y

class LayerScale(nn.Module):
	"""
	LayerScale from CaiT: https://arxiv.org/abs/2103.17239
	Applies a learnable per-channel scaling factor to the input.
	"""
	def __init__(self, dim: int, init_values: float = 1e-5):
		super().__init__()
		self.gamma = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)))

	def forward(self, x):
		return x * self.gamma

class StochasticDepth(nn.Module):
	def __init__(self, p: float):
		"""
		Stochastic Depth (drops entire residual blocks).
		"""
		super().__init__()
		self.p = p
		
	def stochastic_depth(self, x):

		if self.p == 0.0 or not self.training:
			return x
		
		keep_prob = 1 - self.p
		shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, 1, 1, 1)
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_() # binarize
		
		if keep_prob > 0.0:
			random_tensor.div_(keep_prob)
			
		return x * random_tensor

	def forward(self, x):
		return self.stochastic_depth(x)


class GhostModule(nn.Module):
	"""
	Ghost Module for building efficient neural networks.
	Based on the paper: "GhostNet: More Features from Cheap Operations"
	"""
	def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
		super().__init__()
		self.out_channels = out_channels
		init_channels = math.ceil(out_channels / ratio)
		new_channels = init_channels * (ratio - 1)

		# Primary convolution
		self.primary_conv = nn.Sequential(
			nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
			nn.BatchNorm2d(init_channels),
			nn.SiLU(inplace=True) if relu else nn.Sequential(),
		)

		# Cheap operation (depthwise convolution) to generate ghost features
		self.cheap_operation = nn.Sequential(
			nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
			nn.BatchNorm2d(new_channels),
			nn.SiLU(inplace=True) if relu else nn.Sequential(),
		)

	def forward(self, x):
		x1 = self.primary_conv(x)
		x2 = self.cheap_operation(x1)
		out = torch.cat([x1, x2], dim=1)

		return out[:, :self.out_channels, :, :]

	def fuse(self):
		"""Fuse the internal Conv-BN modules."""
		torch.ao.quantization.fuse_modules(self.primary_conv, ['0', '1'], inplace=True)
		torch.ao.quantization.fuse_modules(self.cheap_operation, ['0', '1'], inplace=True)

class ConvBnAct(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1, relu=True):
		super().__init__()
		
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, groups=groups, bias=False)
		self.batch_norm = nn.BatchNorm2d(out_channels)
		self.activation = nn.SiLU() if relu else nn.Identity()
		
	def forward(self, x):
		x = self.conv(x)
		x = self.batch_norm(x)
		x = self.activation(x)
		
		return x

	def fuse(self):
		"""Fuse the conv and bn layers."""
		torch.ao.quantization.fuse_modules(self, ['conv', 'batch_norm'], inplace=True)


class MBConv(nn.Module):
	def __init__(self, in_channels, out_channels, stride, expand_ratio, drop_rate):
		super().__init__()
		self.use_residual = (in_channels == out_channels) and (stride == 1)
		hidden_dim = in_channels * expand_ratio
		
		layers = []

		if expand_ratio != 1:
			layers.append(GhostModule(in_channels, hidden_dim, kernel_size=1, relu=True))
		
		# Depthwise convolution
		layers.extend([
			ConvBnAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
			ECA(hidden_dim),
			GhostModule(hidden_dim, out_channels, kernel_size=1, relu=False),
		])
		
		self.conv = nn.Sequential(*layers)
		self.stochastic_depth = StochasticDepth(drop_rate)
		self.layerscale = LayerScale(out_channels)

	def forward(self, x):
		if self.use_residual:
			return x + self.stochastic_depth(self.layerscale(self.conv(x)))
		else:
			return self.conv(x)


class FusedMBConv(nn.Module):
	def __init__(self, in_channels, out_channels, stride, expand_ratio, drop_rate):
		super().__init__()
		self.use_residual = (in_channels == out_channels) and (stride == 1)
		hidden_dim = in_channels * expand_ratio
		layers = []
		
		# Expansion phase (if expand_ratio > 1)
		if expand_ratio != 1:
			layers.append(GhostModule(in_channels, hidden_dim, kernel_size=3, stride=stride))
		else:
			layers.append(ConvBnAct(in_channels, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=in_channels))

		# Depthwise convolution
		layers.extend([
			ECA(hidden_dim),
			GhostModule(hidden_dim, out_channels, 1, relu=False),
		])
		
		self.conv = nn.Sequential(*layers)
		self.stochastic_depth = StochasticDepth(drop_rate)
		self.layerscale = LayerScale(out_channels)

	def forward(self, x):
		if self.use_residual:
			return x + self.stochastic_depth(self.layerscale(self.conv(x)))
		else:
			return self.conv(x)

class GhostEfficientNetV2(nn.Module):
	def __init__(self, num_classes=6, width_mult=1.0, depth_mult=1.0, drop_rate=0.2, use_arcface=False):
		super().__init__()
		
		self.use_arcface = use_arcface
		
		self.config = [
			(FusedMBConv, 1, 24, 2, 1),
			(FusedMBConv, 4, 48, 4, 2),
			(FusedMBConv, 4, 64, 4, 2),
			(MBConv, 4, 128, 6, 2),
			(MBConv, 6, 160, 9, 1),
			(MBConv, 6, 256, 15, 2)
		]
		out_channels = int(24 * width_mult)
		
		self.stem = nn.Sequential(
			nn.Conv2d(1, out_channels, 3, 2, 1, bias=False), 
			nn.BatchNorm2d(out_channels), 
			nn.SiLU(),
		)
		
		in_channels = out_channels
		blocks = nn.ModuleList()
		total_blocks = sum(int(math.ceil(r * depth_mult)) for _, _, _, r, _ in self.config)
		
		block_idx = 0
		for block, exp, chans, repeats, stride in self.config:
			out_chans = int(chans * width_mult)
			num_repeats = int(math.ceil(repeats * depth_mult))
			
			for i in range(num_repeats):
				s = stride if i == 0 else 1
				dr = drop_rate * block_idx / total_blocks
				blocks.append(block(in_channels, out_chans, s, exp, dr))
				
				in_channels = out_chans
				block_idx += 1
		
		self.pool = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Flatten(),
			nn.Dropout(p=drop_rate, inplace=True),
		)

		self.backbone = nn.Sequential(self.stem, *blocks, self.pool)
		
		if use_arcface:
			self.head = CosineLinear(in_channels, num_classes)
		else:
			self.head = nn.Linear(in_channels, num_classes)

	def forward(self, x, labels=None):
		embs = self.backbone(x)
		logits = self.head(embs)
	
		return logits

	def fuse_model(self):
		if self.training:
			print("Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return
		
		torch.ao.quantization.fuse_modules(self.stem, ['0', '1'], inplace=True)
		
		for m in self.modules():
			if hasattr(m, 'fuse'):
				m.fuse()
		
		print("Model fusion completed successfully.")
