import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.mobilenetv2 import InvertedResidual
from torchvision.ops.misc import ConvBNActivation

from model.utils import ArcMarginProduct

class PannsMobileNetV2ArcFace(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool, use_arcface: bool = False, **arcface_params):
		super().__init__()
		
		weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
		self.cnn = mobilenet_v2(weights=weights)
		
		self._modify_first_conv_layer(in_channels=1, pretrained=pretrained)

		self.use_arcface = use_arcface
		in_features = self.cnn.classifier[1].in_features
		self.cnn.classifier = nn.Identity()

		if self.use_arcface:
			self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else:
			self.head = nn.Linear(in_features, num_classes)
		
	def _modify_first_conv_layer(self, in_channels: int, pretrained: bool):
		"""Modifies the input layer to accept a specified number of input channels."""
		first_conv_module = self.cnn.features[0]
		first_conv = first_conv_module[0]
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
			
		self.cnn.features[0][0] = new_first_conv

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
		if x.dim() == 3:
			x = x.unsqueeze(1)
			
		x = self.cnn.features(x)
		x = F.adaptive_avg_pool2d(x, (1, 1))
		embedding = torch.flatten(x, 1)
		
		if self.use_arcface:
			if self.training:
				if labels is None:
					raise ValueError("Labels are required for ArcFace training.")
				return self.head(embedding, labels)
			else:
				# During inference, ArcFace should output normalized embeddings
				return F.normalize(embedding, p=2, dim=1)
		else:
			return self.head(embedding)

	def fuse_model(self):
		if self.training:
			print("Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return
		
		for m in self.cnn.modules():
			if isinstance(m, ConvBNActivation):
				torch.ao.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)
			elif isinstance(m, InvertedResidual):
				if len(m.conv) > 1 and isinstance(m.conv[-2], nn.Conv2d) and isinstance(m.conv[-1], nn.BatchNorm2d):
					torch.ao.quantization.fuse_modules(m.conv, [str(len(m.conv)-2), str(len(m.conv)-1)], inplace=True)
		
		print("Model fusion completed successfully.")
