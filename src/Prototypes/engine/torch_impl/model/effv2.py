import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from omegaconf import DictConfig

from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf, MBConv, FusedMBConv
from functools import partial

from model.utils import ArcMarginProduct

MODEL_CONFIGS = {
	"efficientnet_v2_s": {"dropout": 0.2},
	"efficientnet_v2_m": {"dropout": 0.3},
	"efficientnet_v2_l": {"dropout": 0.4},
}

class EfficientNetV2ArcFace(EfficientNet):
	"""
	An EfficientNet model extended with ArcFace 
	"""
	def __init__(self, model_name: str, pretrained: bool, num_classes: int, use_arcface: bool, in_channels: int = 1, trainable_blocks: int = 0, **arcface_params):
		if not model_name.startswith('efficientnet_v2'):
			raise ValueError("This class is designed for EfficientNetV2 models.")

		if model_name not in MODEL_CONFIGS:
			raise ValueError(f"Unsupported model: {model_name}. Try one of {list(MODEL_CONFIGS.keys())}")

		inverted_residual_setting, last_channel = _efficientnet_conf(model_name)
		dropout = MODEL_CONFIGS[model_name]["dropout"]

		original_weights_enum = getattr(models, f"EfficientNet_V2_{model_name.split('_')[-1].title()}_Weights")
		temp_num_classes = len(original_weights_enum.DEFAULT.meta["categories"]) if pretrained else num_classes

		super().__init__(
			inverted_residual_setting,
			dropout=dropout,
			last_channel=last_channel,
			num_classes=temp_num_classes
		)
		
		if pretrained:
			self.load_state_dict(original_weights_enum.DEFAULT.get_state_dict(progress=True))

		self.use_arcface = use_arcface
		self._modify_first_conv_layer(in_channels, pretrained)

		in_features = self.classifier[1].in_features
		if self.use_arcface:
			self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else:
			self.head = nn.Linear(in_features, num_classes)
		
		self.classifier = nn.Identity()
		
		self.set_trainable_layers(trainable_blocks)

	def _modify_first_conv_layer(self, in_channels: int, pretrained: bool):
		"""Modifies the input layer to accept a specified number of input channels."""
		first_conv = self.features[0][0]
		original_in_channels = first_conv.in_channels
		if in_channels == original_in_channels:
			return

		new_first_conv = nn.Conv2d(
			in_channels=in_channels, out_channels=first_conv.out_channels,
			kernel_size=first_conv.kernel_size, stride=first_conv.stride,
			padding=first_conv.padding, bias=first_conv.bias is not None
		)

		if pretrained:
			print(f"Adapting pretrained weights of first conv layer from {original_in_channels} to {in_channels} channels.")
			original_weights = first_conv.weight.data
			if original_in_channels == 3 and in_channels == 1:
				new_weights = original_weights.mean(dim=1, keepdim=True)
			else:
				new_weights = original_weights.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / original_in_channels
			new_first_conv.weight.data = new_weights
			
		self.features[0][0] = new_first_conv
		
	def set_trainable_layers(self, trainable_blocks: int = 0):
		"""Freezes or unfreezes model layers for transfer learning."""
		for param in self.parameters():
			param.requires_grad = False
			
		if trainable_blocks == -1:
			for param in self.parameters():
				param.requires_grad = True
			print("Model unfrozen: All layers are now trainable.")
			return

		for param in self.head.parameters():
			param.requires_grad = True
		
		if trainable_blocks > 0:
			num_feature_blocks = len(self.features)
			trainable_blocks = min(trainable_blocks, num_feature_blocks)
			for i in range(num_feature_blocks - trainable_blocks, num_feature_blocks):
				for param in self.features[i].parameters():
					param.requires_grad = True
		
		trainable_msg = "Only the head is trainable."
		if trainable_blocks > 0:
			trainable_msg = f"The head and the last {trainable_blocks} feature blocks are trainable."

		print(f"Model layers configured: {trainable_msg}")

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
		"""Overrides the parent forward method to use the custom head."""
		embedding_vectors = self.features(x)
		embedding_vectors = self.avgpool(embedding_vectors)
		embedding = torch.flatten(embedding_vectors, 1)

		if self.use_arcface:
			if self.training:
				if labels is None:
					raise ValueError("Labels are required for ArcFace training.")
				return self.head(embedding, labels)
			else:
				return F.normalize(embedding, p=2, dim=1)
		else:
			return self.head(embedding)

	def _fuse_conv_bn(self, block: nn.Sequential):
		"""Helper to fuse consecutive Conv2d and BatchNorm2d in a sequential block."""
		fuse_candidates = []
		for i in range(len(block) - 1):
			if isinstance(block[i], nn.Conv2d) and isinstance(block[i+1], nn.BatchNorm2d):
				fuse_candidates.append([str(i), str(i+1)])
		
		if fuse_candidates:
			torch.ao.quantization.fuse_modules(block, fuse_candidates, inplace=True)
			
	def fuse_model(self):
		"""
		Fuses Conv2d and BatchNorm2d layers for faster inference.
		"""

		if self.training:
			print("⚠️ Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return

		self._fuse_conv_bn(self.features[0])

		for module in self.modules():
			if isinstance(module, (MBConv, FusedMBConv)):
				self._fuse_conv_bn(module.block)
				
		print("Model fusion completed successfully.")
