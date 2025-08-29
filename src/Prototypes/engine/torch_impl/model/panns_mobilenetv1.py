import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from model.utils import ArcMarginProduct

PRETRAINED_URL = "https://zenodo.org/record/3987831/files/MobileNetV1_mAP%3D0.389.pth"

def init_layer(layer):
	"""Initialize a Linear or Convolutional layer."""
	nn.init.xavier_uniform_(layer.weight)
	if hasattr(layer, 'bias') and layer.bias is not None:
		layer.bias.data.fill_(0.)

def init_bn(bn):
	"""Initialize a Batchnorm layer."""
	bn.bias.data.fill_(0.)
	bn.weight.data.fill_(1.)

def conv_bn_v1(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True)
	)

def conv_dw_v1(inp, oup, stride):
	return nn.Sequential(
		# Depthwise convolution
		nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
		nn.BatchNorm2d(inp),
		nn.ReLU(inplace=True),

		# Pointwise convolution
		nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True),
	)

class MobileNetV1(nn.Module):
	def __init__(self, classes_num):
		super(MobileNetV1, self).__init__()
		self.features = nn.Sequential(
			conv_bn_v1(1, 32, 2), conv_dw_v1(32, 64, 1), conv_dw_v1(64, 128, 2),
			conv_dw_v1(128, 128, 1), conv_dw_v1(128, 256, 2), conv_dw_v1(256, 256, 1),
			conv_dw_v1(256, 512, 2), conv_dw_v1(512, 512, 1), conv_dw_v1(512, 512, 1),
			conv_dw_v1(512, 512, 1), conv_dw_v1(512, 512, 1), conv_dw_v1(512, 512, 1),
			conv_dw_v1(512, 1024, 2), conv_dw_v1(1024, 1024, 1)
		)
		self.fc = nn.Linear(1024, 1024, bias=True)
		self.fc_audioset = nn.Linear(1024, classes_num, bias=True)
		self.init_weights()

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init_layer(m)
			elif isinstance(m, nn.BatchNorm2d):
				init_bn(m)

		init_layer(self.fc)
		init_layer(self.fc_audioset)

	def forward(self, x):
		x = self.features(x)

		x = F.avg_pool2d(x, (x.size(2), x.size(3)))
		x = x.view(x.size(0), -1)
		x = F.dropout(x, p=0.2, training=self.training)
		x = self.fc(x)

		embedding = F.dropout(x, p=0.2, training=self.training)
		clipwise_output = self.fc_audioset(x)

		return clipwise_output, embedding

class PannsMobileNetV1ArcFace(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool, use_arcface: bool = False, **arcface_params):
		super().__init__()
		PRETRAINED_URL = "https://zenodo.org/record/3987831/files/MobileNetV1_mAP%3D0.389.pth"
		original_classes = 527
		self.cnn = MobileNetV1(classes_num=original_classes)

		if pretrained:
			print("Loading pretrained MobileNetV1 weights.")
			state_dict = load_state_dict_from_url(PRETRAINED_URL, progress=True)['model']
			for key in ['spectrogram_extractor.stft.conv_real.weight', 'spectrogram_extractor.stft.conv_imag.weight', 'logmel_extractor.melW', 'bn0.weight', 'bn0.bias', 'bn0.running_mean', 'bn0.running_var', 'bn0.num_batches_tracked']:
				state_dict.pop(key, None)
			self.cnn.load_state_dict(state_dict)
		self.use_arcface = use_arcface

		in_features = self.cnn.fc_audioset.in_features
		if self.use_arcface:
			self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else:
			self.head = nn.Linear(in_features, num_classes)

		self.cnn.fc_audioset = nn.Identity()

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
		if x.dim() == 3: x = x.unsqueeze(1)
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
			if isinstance(module, nn.Sequential):
				if len(module) == 3:
					torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
				elif len(module) == 6:
					torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
					torch.ao.quantization.fuse_modules(module, ['3', '4', '5'], inplace=True)

		print("Model fusion completed successfully.")
