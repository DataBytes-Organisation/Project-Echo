import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import math

from model.utils import ArcMarginProduct

PRETRAINED_URL = "https://zenodo.org/record/3987831/files/MobileNetV2_mAP%3D0.383.pth"

def conv_bn(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)

def conv_1x1_bn(inp, oup):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU6(inplace=True)
	)

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]

		hidden_dim = round(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup

		if expand_ratio == 1:
			self.conv = nn.Sequential(
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
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
				nn.ReLU6(inplace=True),
				# dw
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.ReLU6(inplace=True),
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
	def __init__(self, classes_num=1000, width_mult=1.):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel = 32
		last_channel = 1280
		interverted_residual_setting = [
			# t, c, n, s
			[1, 16, 1, 1],
			[6, 24, 2, 2],
			[6, 32, 3, 2],
			[6, 64, 4, 2],
			[6, 96, 3, 1],
			[6, 160, 3, 2],
			[6, 320, 1, 1],
		]

		# building first layer
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn(1, input_channel, 2)]
		
		# building inverted residual blocks
		for t, c, n, s in interverted_residual_setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				if i == 0:
					self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
				else:
					self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
				input_channel = output_channel
		
		# building last several layers
		self.features.append(conv_1x1_bn(input_channel, self.last_channel))
		# make it nn.Sequential
		self.features = nn.Sequential(*self.features)

		# building classifier
		self.fc = nn.Linear(self.last_channel, self.last_channel)
		self.fc_audioset = nn.Linear(self.last_channel, classes_num)

		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.mean(3).mean(2)
		x = self.fc(x)
		embedding = x
		x = self.fc_audioset(x)
		return x, embedding

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class PannsMobileNetV2ArcFace(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool, use_arcface: bool = False, **arcface_params):
		super().__init__()
		
		original_classes = 527  # AudioSet has 527 classes
		self.cnn = MobileNetV2(classes_num=original_classes)
		
		if pretrained:
			print("Loading pretrained MobileNetV2 weights.")
			state_dict = load_state_dict_from_url(PRETRAINED_URL, progress=True)['model']
			
			state_dict.pop('spectrogram_extractor.stft.conv_real.weight')
			state_dict.pop('spectrogram_extractor.stft.conv_imag.weight')
			state_dict.pop('logmel_extractor.melW')

			self.cnn.load_state_dict(state_dict)

		self.use_arcface = use_arcface
		in_features = self.cnn.fc_audioset.in_features
		self.cnn.fc_audioset = nn.Identity()

		if self.use_arcface:
			self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else:
			self.head = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
		if x.dim() == 3:
			x = x.unsqueeze(1)
			
		_, embedding = self.cnn(x)
		
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
		
		for module in self.cnn.features:
			if isinstance(module, nn.Sequential) and len(module) == 3:
				torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
			elif isinstance(module, InvertedResidual):
				if hasattr(module, 'conv'):
					if len(module.conv) == 8:
						torch.ao.quantization.fuse_modules(module.conv, ['0', '1', '2'], inplace=True)
						torch.ao.quantization.fuse_modules(module.conv, ['3', '4', '5'], inplace=True)
						torch.ao.quantization.fuse_modules(module.conv, ['6', '7'], inplace=True)
					elif len(module.conv) == 5:
						torch.ao.quantization.fuse_modules(module.conv, ['0', '1', '2'], inplace=True)
						torch.ao.quantization.fuse_modules(module.conv, ['3', '4'], inplace=True)
		
		print("Model fusion completed successfully.")
