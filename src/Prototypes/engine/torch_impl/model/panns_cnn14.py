import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import re

from model.utils import ArcMarginProduct

# Pretrained weights for Cnn14, trained on AudioSet with 32kHz audio.
PRETRAINED_URL = "https://zenodo.org/record/3987831/files/Cnn14_32k_mAP%3D0.431.pth"

def init_layer(layer):
	"""Initialize a Linear or Convolutional layer."""
	nn.init.xavier_uniform_(layer.weight)
	if hasattr(layer, 'bias') and layer.bias is not None:
		layer.bias.data.fill_(0.)

def init_bn(bn):
	"""Initialize a Batchnorm layer."""
	bn.bias.data.fill_(0.)
	bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvBlock, self).__init__()
		self.seq1 = nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)
		self.seq2 = nn.Sequential(
			nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()
		)

		init_layer(self.seq1[0])
		init_layer(self.seq2[0])
		init_bn(self.seq1[1])
		init_bn(self.seq2[1])

	def forward(self, x, pool_size=(2, 2), pool_type='avg'):
		x = self.seq1(x)
		x = self.seq2(x)
		if pool_type == 'max':
			x = F.max_pool2d(x, kernel_size=pool_size)
		elif pool_type == 'avg':
			x = F.avg_pool2d(x, kernel_size=pool_size)
		return x

class Cnn14(nn.Module):
	def __init__(self, classes_num):
		super(Cnn14, self).__init__()
		self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
		self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
		self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
		self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
		self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
		self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

		self.fc1 = nn.Linear(2048, 2048, bias=True)
		self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

		init_layer(self.fc1)
		init_layer(self.fc_audioset)

	def forward(self, x):
		x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
		x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
		x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
		x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
		x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
		x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

		x = torch.mean(x, dim=3)
		(x1, _) = torch.max(x, dim=2)
		x2 = torch.mean(x, dim=2)
		x = x1 + x2
		x = F.dropout(x, p=0.5, training=self.training)
		x = F.relu_(self.fc1(x))
		embedding = F.dropout(x, p=0.5, training=self.training)

		return self.fc_audioset(embedding), embedding

class PannsCNN14ArcFace(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool, use_arcface: bool = False, **arcface_params):
		super().__init__()
		PRETRAINED_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth"
		original_classes = 527
		self.cnn = Cnn14(classes_num=original_classes)

		if pretrained:
			print("Loading pretrained Cnn14 weights.")
			state_dict = load_state_dict_from_url(PRETRAINED_URL, progress=True)['model']
			new_state_dict = {}
			for k, v in state_dict.items():
				new_k = k
				if 'conv_block' in k:
					new_k = re.sub(r'\.conv1\.', '.seq1.0.', new_k)
					new_k = re.sub(r'\.bn1\.',	 '.seq1.1.', new_k)
					new_k = re.sub(r'\.conv2\.', '.seq2.0.', new_k)
					new_k = re.sub(r'\.bn2\.',	 '.seq2.1.', new_k)
				new_state_dict[new_k] = v
			self.cnn.load_state_dict(new_state_dict, strict=False)

		self.use_arcface = use_arcface
		in_features = self.cnn.fc_audioset.in_features
		if self.use_arcface:
			self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else:
			self.head = nn.Linear(in_features, num_classes)
		self.cnn.fc_audioset = nn.Identity()

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
				return F.normalize(embedding, p=2, dim=1)
		else:
			return self.head(embedding)

	def fuse_model(self):
		if self.training:
			print("Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return

		for module in self.modules():
			if isinstance(module, ConvBlock):
				torch.ao.quantization.fuse_modules(module.seq1, ['0', '1', '2'], inplace=True)
				torch.ao.quantization.fuse_modules(module.seq2, ['0', '1', '2'], inplace=True)

		print("Model fusion completed successfully.")
