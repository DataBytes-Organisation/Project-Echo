import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math

from model.utils import ArcMarginProduct

PRETRAINED_URL = "https://zenodo.org/record/3987831/files/MobileNetV2_mAP%3D0.383.pth"

class ConvBnReLU6(ConvBn2d):
	def __init__(self, conv, bn):
		super().__init__(conv, bn)
		self.relu6 = nn.ReLU6(inplace=True)

	def forward(self, x):
		x = super(ConvBnReLU6, self).forward(x)
		x = self.relu6(x)
		return x

def fuse_conv_bn_relu6(conv, bn, relu6):
	assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and \
		isinstance(relu6, nn.ReLU6), \
		"Expecting (Conv2d, BatchNorm2d, ReLU6) as input."

	return ConvBnReLU6(conv, bn)

def conv_bn_v2(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
		nn.AvgPool2d(stride),
		nn.BatchNorm2d(oup),
		# nn.ReLU6(inplace=True),
		nn.ReLU(inplace=True),
	)

def conv_1x1_bn_v2(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		# nn.ReLU6(inplace=True),
		nn.ReLU(inplace=True),
	)

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super().__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.AvgPool2d(stride),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				# pw
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
				nn.BatchNorm2d(hidden_dim),
				# nn.ReLU6(inplace=True),
				nn.ReLU(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
				nn.AvgPool2d(stride),
				nn.BatchNorm2d(hidden_dim),
				# nn.ReLU6(inplace=True),
				nn.ReLU(inplace=True),
				# pw-linear
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
				nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect:
			return x + self.conv(x)
		else:
			return self.conv(x)

class MobileNetV2(nn.Module):
	def __init__(self, classes_num=527, **kwargs):
		super(MobileNetV2, self).__init__()

		num_features = kwargs.get('num_features', 64)
		embedding_size = kwargs.get('embedding_size', 1280)

		self.bn0 = nn.BatchNorm2d(num_features)

		width_mult = 1.
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		inverted_residual_setting = [
			# t, c, n, s
			[1, 16, 1, 1],
			[6, 24, 2, 2],
			[6, 32, 3, 2],
			[6, 64, 4, 2],
			[6, 96, 3, 2],
			[6, 160, 3, 1],
			[6, 320, 1, 1],
		]

		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn_v2(1, input_channel, (2, 2))]

		for t, c, n, s in inverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				stride = s if i == 0 else 1
				self.features.append(
					block(input_channel, output_channel, stride, expand_ratio=t))
				input_channel = output_channel

		self.features.append(conv_1x1_bn_v2(input_channel, self.last_channel))
		self.features = nn.Sequential(*self.features)

		self.fc1 = nn.Linear(1280, embedding_size, bias=True)
		self.fc_audioset = nn.Linear(embedding_size, classes_num, bias=True)

		self.init_weights()

	def init_weights(self):
		init_bn(self.bn0)
		init_layer(self.fc1)
		init_layer(self.fc_audioset)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		x = x.transpose(1, 2)
		x = self.bn0(x)
		x = x.transpose(1, 2)

		x = self.features(x)
		x = torch.mean(x, dim=3)

		(x1, _) = torch.max(x, dim=2)
		x2 = torch.mean(x, dim=2)
		x = x1 + x2

		x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu_(self.fc1(x))
		embedding = F.dropout(x, p=0.5, training=self.training)
		clipwise_output = self.fc_audioset(embedding)

		return clipwise_output, embedding

class PannsMobileNetV2ArcFace(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool, use_arcface: bool = False, **arcface_params):
		super().__init__()
		PRETRAINED_URL = "https://zenodo.org/record/3987831/files/MobileNetV2_mAP%3D0.383.pth"
		original_classes = 527
		self.cnn = MobileNetV2(classes_num=original_classes, num_features=64, embedding_size=1024)

		if pretrained:
			print("Loading pretrained MobileNetV2 weights.")
			state_dict = load_state_dict_from_url(PRETRAINED_URL, progress=True)['model']
			for key in ['spectrogram_extractor.stft.conv_real.weight', 'spectrogram_extractor.stft.conv_imag.weight', 'logmel_extractor.melW']:
				state_dict.pop(key, None)
			self.cnn.load_state_dict(state_dict)

		self.use_arcface = use_arcface
		in_features = self.cnn.fc_audioset.in_features
		self.cnn.fc_audioset = nn.Identity()

		if self.use_arcface:
			self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else:
			self.head = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
		# if x.dim() == 3:
		#	 x = x.unsqueeze(1)

		_, embedding = self.cnn(x)
		if self.use_arcface:
			if self.training:
				if labels is None:
					raise ValueError("Labels are required for ArcFace training.")
				return self.head(embedding, labels)
			else:
				return F.normalize(embedding, p=2, dim=1)
		else:
			return self.head(embedding)

	def fuse_model(self):
		if self.training:
			print("Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return

		for module in self.cnn.features:
			if isinstance(module, nn.Sequential) and len(module) == 3:
				torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
			elif isinstance(module, InvertedResidual) and hasattr(module, 'conv'):
				if len(module.conv) == 8:
					torch.ao.quantization.fuse_modules(module.conv, ['0', '1', '2'], inplace=True)
					torch.ao.quantization.fuse_modules(module.conv, ['3', '4', '5'], inplace=True)
					torch.ao.quantization.fuse_modules(module.conv, ['6', '7'], inplace=True)
				elif len(module.conv) == 5:
					torch.ao.quantization.fuse_modules(module.conv, ['0', '1', '2'], inplace=True)
					torch.ao.quantization.fuse_modules(module.conv, ['3', '4'], inplace=True)

		print("Model fusion completed successfully.")
