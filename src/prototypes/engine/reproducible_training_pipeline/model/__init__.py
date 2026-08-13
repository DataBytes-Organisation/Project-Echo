import io

import torch
import torch.nn as nn
import torchvision.models as models

from omegaconf import DictConfig

from model.effv2 import EfficientNetV2ArcFace
from model.ghost_effv2 import GhostEfficientNetV2
from model.panns_cnn14 import PannsCNN14ArcFace
from model.panns_mobilenetv1 import PannsMobileNetV1ArcFace
from model.panns_mobilenetv2 import PannsMobileNetV2ArcFace
from model.quant import prepare_qat_fx, prepare_post_static_quantize_fx, convert_fx

from enum import Enum


class RMSNorm2d(nn.Module):
	"""
	Standard RMSNorm (no mean subtraction) adapted for 4D CNN input (B, C, H, W).
	Normalizes across (C, H, W) dimensions for each sample.
	"""

	def __init__(self, channels: int, eps: float = 1e-6, add_unit_offset: bool = True):
		super().__init__()
		self.eps = eps
		self.add_unit_offset = add_unit_offset
		self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

	def _norm(self, x):
		# Calculate RMS across (C, H, W).
		# Mean operation is over dimensions 1, 2, 3 (Channel, Height, Width)
		# Sum of x.pow(2) divided by the total number of elements used for the mean (C*H*W)

		# Calculate the mean of the squared input over the normalization axis
		# dim=[1, 2, 3] ensures that the statistics are calculated per image.
		mean_squared = x.pow(2).mean(dim=[1, 2, 3], keepdim=True)

		# Calculate 1/sqrt(mean_squared + eps)
		rsqrt_val = torch.rsqrt(mean_squared + self.eps)
		return x * rsqrt_val

	def forward(self, x):
		output = self._norm(x.float())

		# Apply learned weight (scale)
		if self.add_unit_offset:
			output = output * (1 + self.weight.float())
		else:
			output = output * self.weight.float()

		return output.type_as(x)


class Model(nn.Module):
	def __init__(self, cfg: DictConfig):
		super().__init__()
		model_params = cfg.model.params
		norm_choice = cfg.model.norm_choice

		if model_params.model_name.startswith("efficientnet_v2"):
			self.model = EfficientNetV2ArcFace(
				model_name=model_params.model_name,
				pretrained=model_params.pretrained,
				num_classes=model_params.num_classes,
				use_arcface=model_params.get("use_arcface", False),
				trainable_blocks=model_params.get("trainable_blocks", 0),
			)
		elif model_params.model_name.startswith("ghost_efficientnet_v2"):
			self.model = GhostEfficientNetV2(
				num_classes=model_params.num_classes,
				width_mult=model_params.get("width_mult", 0.75),
				depth_mult=model_params.get("depth_mult", 0.5),
				drop_rate=model_params.get("drop_rate", 0.2),
				use_arcface=model_params.get("use_arcface", False),
			)
		elif model_params.model_name.startswith("panns_cnn14"):
			self.model = PannsCNN14ArcFace(
				classes_num=model_params.classes_num,
				pretrained=model_params.get("pretrained", True),
				use_arcface=model_params.get("use_arcface", False),
			)
		elif model_params.model_name.startswith("panns_mobilenetv1"):
			self.model = PannsMobileNetV1ArcFace(
				classes_num=model_params.classes_num,
				pretrained=model_params.get("pretrained", True),
				use_arcface=model_params.get("use_arcface", False),
			)
		elif model_params.model_name.startswith("panns_mobilenetv2"):
			self.model = PannsMobileNetV2ArcFace(
				num_classes=model_params.num_classes,
				pretrained=model_params.get("pretrained", True),
				use_arcface=model_params.get("use_arcface", False),
			)
		else:
			raise ValueError(f"Model '{model_params.model_name}' not supported.")

		if norm_choice == "freeze_bn":
			self.freeze_batch_norm()
		elif norm_choice == "swap_rms_norm":
			self.swap_batch_norm(RMSNorm2d)
		elif norm_choice == "keep_bn":
			# Do nothing
			pass

		self.use_qat = model_params.get("use_qat", False)
		if self.use_qat:
			self.model.eval()
			self.model.fuse_model()
			self.model = prepare_qat_fx(self.model)

	def freeze_batch_norm(self):
		"""
		Sets all BatchNorm layers in a model to evaluation mode (freeze statistics)
		and freezes their weights (gamma/beta).
		"""
		print("Applying: FREEZE_BN. Setting all BatchNorm layers to eval() mode.")
		for module in self.model.modules():
			if isinstance(module, nn.BatchNorm2d):
				module.eval()
				for param in module.parameters():
					param.requires_grad = False

	def swap_batch_norm(self, target_norm: type, **kwargs):
		"""
		Replaces all BatchNorm2d layers in the backbone with the target normalization layer.
		"""
		norm_name = target_norm.__name__
		print(f"Applying: SWAP_NORM. Replacing BatchNorm2d with {norm_name}.")

		def replace_module(module: nn.Module):
			for name, child in module.named_children():
				if isinstance(child, nn.BatchNorm2d):
					num_features = child.num_features

					if target_norm == nn.GroupNorm:
						# For GroupNorm, must specify num_groups and num_channels
						# Setting num_groups=1 is the CNN-friendly equivalent of LayerNorm
						new_norm = target_norm(num_groups=1, num_channels=num_features, **kwargs)
					elif target_norm == RMSNorm2d:
						new_norm = target_norm(num_features, **kwargs)
					else:
						raise ValueError(f"Unsupported target norm: {norm_name}")

					# Replace the module in the parent container
					setattr(module, name, new_norm)
				else:
					replace_module(child)  # Recursively apply to children

		replace_module(self.model)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)

	def load_state_dict(self, state_dict, strict=True):
		return self.model.load_state_dict(state_dict, strict=strict)

	def state_dict(self, *args, **kwargs):
		return self.model.state_dict(*args, **kwargs)

	def quantise(self):
		if not self.use_qat:
			print("Warning: quantise() called, but model was not trained with QAT. Only fusing modules.")
			self.model.eval()
			self.model.fuse_model()
			return

		self.model = convert_fx(self.model)

	def summary(self):
		"""
		Prints a summary of the model, including the number of parameters
		and its estimated size on disk.
		"""

		total_params = sum(p.numel() for p in self.model.parameters())
		trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

		buffer = io.BytesIO()
		torch.save(self.state_dict(), buffer)
		size_mb = buffer.getbuffer().nbytes / (1024**2)

		print("=" * 65)
		print(f"Model Summary: {self.model.__class__.__name__}")
		print("-" * 65)
		print(f"{'Total Parameters':<30}: {total_params:,}")
		print(f"{'Trainable Parameters':<30}: {trainable_params:,}")
		print(f"{'Non-trainable Parameters':<30}: {total_params - trainable_params:,}")
		print(f"{'Estimated Model Size (MB)':<30}: {size_mb:.2f}")
		print("=" * 65)
