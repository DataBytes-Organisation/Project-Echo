import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.hub import load_state_dict_from_url
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import torch.ao.quantization as quant
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx as prepare_qat_fx_

import os
from pathlib import Path
import librosa
import numpy as np
import hashlib
import diskcache as dc
import math
import re
import copy
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from torchvision import transforms
import torchvision.models as models
from torchvision.models.efficientnet import EfficientNet, _efficientnet_conf, MBConv, FusedMBConv
from functools import partial
import umap
import matplotlib.pyplot as plt

# --- Configuration ---
class Config:
	"""
	Configuration class to replace Hydra.
	All settings for the training run are defined here.
	"""
	def __init__(self):
		self.system = self.System()
		self.data = self.Data()
		self.training = self.Training()
		self.model = self.Model()
		self.augmentations = self.Augmentations()

	class System:
		audio_data_directory = 'REPLACE_WITH_YOUR_DATA_DIRECTORY'
		use_disk_cache = True
		cache_directory = './cache'

	class Data:
		sample_rate = 32000
		audio_clip_duration = 5.0
		train_split = 0.8
		val_split = 0.2
		n_fft = 1024
		hop_length = 320
		n_mels = 64
		fmin = 50
		fmax = 14000
		top_db = 80.0
		num_classes = None

	class Training:
		def __init__(self):
			self.seed = 0
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.batch_size = 32
			self.num_workers = 4
			self.epochs = 50
			self.metric_mode = 'min'  # 'min' for loss, 'max' for accuracy
			self.dtype = 'bfloat16'
			
			self.criterion = 'CrossEntropyLoss'
			self.optimizer = 'AdamW'
			self.scheduler = 'ReduceLROnPlateau'

			self.criterion_params = self.CriterionParams()
			self.optimizer_params = self.OptimizerParams()
			self.scheduler_params = self.SchedulerParams()
			self.distillation = self.Distillation()

		class CriterionParams:
			pass
		
		class OptimizerParams:
			def __init__(self):
				self.lr = 1e-3
		
		class SchedulerParams:
			def __init__(self):
				self.mode = 'min'
				self.factor = 0.1
				self.patience = 5
		
		class Distillation:
			def __init__(self):
				self.enabled = False
				self.teacher_model_path = None
				self.criterion = 'KLDivLoss'
				self.alpha = 0.5
				self.temperature = 3.0

	class Model:
		def __init__(self):
			self.name = 'efficientnetv2'
			self.params = self.Params()

		class Params:
			def __init__(self):
				self.model_name = 'efficientnet_v2_s'
				self.pretrained = True
				self.num_classes = None
				self.use_arcface = False
				self.s = 30.0
				self.m = 0.50
				self.trainable_blocks = 0
				self.use_qat = False

	class Augmentations:
		def __init__(self):
			self.audio = Compose([
				AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.2),
				TimeStretch(min_rate=0.8, max_rate=1.25, p=0.2),
				PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
				Shift(min_fraction=-0.5, max_fraction=0.5, p=0.2),
			])
			
			self.image = TorchCompose([
				SpecAugment(
					p=0.5,
					freq_mask_param=30,
					time_mask_param=80,
					n_freq_mask=2,
					n_time_mask=2
				)
			])

		def get_audio_transforms(self):
			return self.audio
		
		def get_image_transforms(self):
			return self.image

# --- Model Utilities (from model/utils.py) ---
class ArcMarginProduct(nn.Module):
	def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
		super().__init__()

		self.in_features = in_features
		self.out_features = out_features

		self.s = s
		self.m = m

		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)

		self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

	def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

		phi = cosine * self.cos_m - sine * self.sin_m
		phi = torch.where(cosine > self.th, phi, cosine - self.s * self.sin_m * self.m)

		one_hot = torch.zeros(cosine.size(), device=input.device)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)

		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output *= self.s

		return output

def get_all_embeddings(model, dataloader, device):
	"""
	Extracts embeddings for all samples in a dataloader using a given model.
	Assumes model is on the correct device and in eval mode.
	"""
	model.eval()
	all_embeddings = []
	all_labels = []

	with torch.no_grad():
		for inputs, labels in tqdm(dataloader, desc="Generating embeddings"):
			inputs = inputs.to(device)
			# This assumes that model in eval mode returns embeddings.
			# This is true for the ArcFace models in the codebase when not training.
			outputs = model(inputs)
			all_embeddings.append(outputs.cpu().numpy())
			all_labels.append(labels.cpu().numpy())

	all_embeddings = np.concatenate(all_embeddings, axis=0)
	all_labels = np.concatenate(all_labels, axis=0)
	return all_embeddings, all_labels

def plot_class_distribution_umap(embeddings, labels, class_names, title='UMAP projection of class embeddings', save_path=None):
	"""
	Generates and displays/saves a UMAP plot of embeddings, colored by class.
	"""
	print("Running UMAP dimensionality reduction...")
	reducer = umap.UMAP(
		n_neighbors=15,
		min_dist=0.1,
		n_components=2,
		random_state=42,
		metric='cosine' # Cosine is good for normalized embeddings like those from ArcFace
	)
	embedding_2d = reducer.fit_transform(embeddings)

	plt.figure(figsize=(14, 12))
	unique_labels = np.unique(labels)
	
	# Use a color map for better visualization of many classes
	colors = plt.cm.get_cmap('jet', len(unique_labels))

	for i, label_idx in enumerate(unique_labels):
		class_indices = np.where(labels == label_idx)[0]
		plt.scatter(
			embedding_2d[class_indices, 0],
			embedding_2d[class_indices, 1],
			color=colors(i),
			label=class_names[label_idx],
			alpha=0.7,
			s=10 # smaller points for clarity with many points
		)

	plt.title(title, fontsize=16)
	plt.xlabel('UMAP Dimension 1', fontsize=12)
	plt.ylabel('UMAP Dimension 2', fontsize=12)
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=10)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend
	
	if save_path:
		plt.savefig(save_path, bbox_inches='tight', dpi=300)
		print(f"Plot saved to {save_path}")
	
	plt.show()

# --- Augmentations (from augment.py) ---
class SpecAugment(nn.Module):
	"""
	Spectrogram augmentation module.
	Reference: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
	(https://arxiv.org/abs/1904.08779)
	"""
	def __init__(self, p=0.5, freq_mask_param=30, time_mask_param=80, n_freq_mask=2, n_time_mask=2):
		super().__init__()
		self.p = p
		self.freq_mask_param = freq_mask_param
		self.time_mask_param = time_mask_param
		self.n_freq_mask = n_freq_mask
		self.n_time_mask = n_time_mask

	def forward(self, x):
		if random.random() > self.p:
			return x
		
		# x is (C, F, T)
		for _ in range(self.n_freq_mask):
			f = random.randint(0, self.freq_mask_param)
			f0 = random.randint(0, x.shape[1] - f)
			x[:, f0:f0+f, :] = 0
		
		for _ in range(self.n_time_mask):
			t = random.randint(0, self.time_mask_param)
			t0 = random.randint(0, x.shape[2] - t)
			x[:, :, t0:t0+t] = 0
			
		return x

# --- Models ---

# --- PANNS CNN14 (from model/panns_cnn14.py) ---
def init_layer(layer):
	nn.init.xavier_uniform_(layer.weight)
	if hasattr(layer, 'bias') and layer.bias is not None:
		layer.bias.data.fill_(0.)

def init_bn(bn):
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
		PRETRAINED_URL = "https://zenodo.org/record/3987831/files/Cnn14_32k_mAP%3D0.431.pth"
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
					new_k = re.sub(r'\.bn1\.',   '.seq1.1.', new_k)
					new_k = re.sub(r'\.conv2\.', '.seq2.0.', new_k)
					new_k = re.sub(r'\.bn2\.',   '.seq2.1.', new_k)
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

# --- PANNS MobileNetV1 (from model/panns_mobilenetv1.py) ---
def conv_bn_v1(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
		nn.BatchNorm2d(oup),
		nn.ReLU(inplace=True)
	)

def conv_dw_v1(inp, oup, stride):
	return nn.Sequential(
		nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
		nn.BatchNorm2d(inp),
		nn.ReLU(inplace=True),
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
			if isinstance(m, nn.Conv2d): init_layer(m)
			elif isinstance(m, nn.BatchNorm2d): init_bn(m)
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
				if labels is None: raise ValueError("Labels are required for ArcFace training.")
				return self.head(embedding, labels)
			else: return F.normalize(embedding, p=2, dim=1)
		else: return self.head(embedding)

	def fuse_model(self):
		if self.training:
			print("Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return
		for module in self.cnn.features:
			if isinstance(module, nn.Sequential):
				if len(module) == 3: torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
				elif len(module) == 6:
					torch.ao.quantization.fuse_modules(module, ['0', '1', '2'], inplace=True)
					torch.ao.quantization.fuse_modules(module, ['3', '4', '5'], inplace=True)
		print("Model fusion completed successfully.")

# --- PANNS MobileNetV2 (from model/panns_mobilenetv2.py) ---
def conv_bn_v2(inp, oup, stride):
	return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

def conv_1x1_bn_v2(inp, oup):
	return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super(InvertedResidual, self).__init__()
		self.stride = stride
		assert stride in [1, 2]
		hidden_dim = round(inp * expand_ratio)
		self.use_res_connect = self.stride == 1 and inp == oup
		if expand_ratio == 1:
			self.conv = nn.Sequential(
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
			)
		else:
			self.conv = nn.Sequential(
				nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
				nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True),
				nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup),
			)

	def forward(self, x):
		if self.use_res_connect: return x + self.conv(x)
		else: return self.conv(x)

class MobileNetV2(nn.Module):
	def __init__(self, classes_num=1000, width_mult=1.):
		super(MobileNetV2, self).__init__()
		block = InvertedResidual
		input_channel, last_channel = 32, 1280
		setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]]
		input_channel = int(input_channel * width_mult)
		self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
		self.features = [conv_bn_v2(1, input_channel, 2)]
		for t, c, n, s in setting:
			output_channel = int(c * width_mult)
			for i in range(n):
				self.features.append(block(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t))
				input_channel = output_channel
		self.features.append(conv_1x1_bn_v2(input_channel, self.last_channel))
		self.features = nn.Sequential(*self.features)
		self.fc = nn.Linear(self.last_channel, self.last_channel)
		self.fc_audioset = nn.Linear(self.last_channel, classes_num)
		self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.mean(3).mean(2)
		x = self.fc(x)
		return self.fc_audioset(x), x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None: m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

class PannsMobileNetV2ArcFace(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool, use_arcface: bool = False, **arcface_params):
		super().__init__()
		PRETRAINED_URL = "https://zenodo.org/record/3987831/files/MobileNetV2_mAP%3D0.383.pth"
		original_classes = 527
		self.cnn = MobileNetV2(classes_num=original_classes)
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
		if x.dim() == 3: x = x.unsqueeze(1)
		_, embedding = self.cnn(x)
		if self.use_arcface:
			if self.training:
				if labels is None: raise ValueError("Labels are required for ArcFace training.")
				return self.head(embedding, labels)
			else: return F.normalize(embedding, p=2, dim=1)
		else: return self.head(embedding)

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

# --- EfficientNetV2 (from model/effv2.py) ---
MODEL_CONFIGS = {"efficientnet_v2_s": {"dropout": 0.2}, "efficientnet_v2_m": {"dropout": 0.3}, "efficientnet_v2_l": {"dropout": 0.4}}

class EfficientNetV2ArcFace(EfficientNet):
	def __init__(self, model_name: str, pretrained: bool, num_classes: int, use_arcface: bool, in_channels: int = 1, trainable_blocks: int = 0, **arcface_params):
		if not model_name.startswith('efficientnet_v2'): raise ValueError("This class is for EfficientNetV2 models.")
		if model_name not in MODEL_CONFIGS: raise ValueError(f"Unsupported model: {model_name}. Try one of {list(MODEL_CONFIGS.keys())}")
		inverted_residual_setting, last_channel = _efficientnet_conf(model_name)
		dropout = MODEL_CONFIGS[model_name]["dropout"]
		original_weights_enum = getattr(models, f"EfficientNet_V2_{model_name.split('_')[-1].title()}_Weights")
		temp_num_classes = len(original_weights_enum.DEFAULT.meta["categories"]) if pretrained else num_classes
		super().__init__(inverted_residual_setting, dropout=dropout, last_channel=last_channel, num_classes=temp_num_classes)
		if pretrained: self.load_state_dict(original_weights_enum.DEFAULT.get_state_dict(progress=True))
		self.use_arcface = use_arcface
		self._modify_first_conv_layer(in_channels, pretrained)
		in_features = self.classifier[1].in_features
		if self.use_arcface: self.head = ArcMarginProduct(in_features, num_classes, **arcface_params)
		else: self.head = nn.Linear(in_features, num_classes)
		self.classifier = nn.Identity()
		self.set_trainable_layers(trainable_blocks)

	def _modify_first_conv_layer(self, in_channels: int, pretrained: bool):
		first_conv = self.features[0][0]
		original_in_channels = first_conv.in_channels
		if in_channels == original_in_channels: return
		new_first_conv = nn.Conv2d(in_channels=in_channels, out_channels=first_conv.out_channels, kernel_size=first_conv.kernel_size, stride=first_conv.stride, padding=first_conv.padding, bias=first_conv.bias is not None)
		if pretrained:
			print(f"Adapting pretrained weights of first conv layer from {original_in_channels} to {in_channels} channels.")
			original_weights = first_conv.weight.data
			new_weights = original_weights.mean(dim=1, keepdim=True) if original_in_channels == 3 and in_channels == 1 else original_weights.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / original_in_channels
			new_first_conv.weight.data = new_weights
		self.features[0][0] = new_first_conv

	def set_trainable_layers(self, trainable_blocks: int = 0):
		for param in self.parameters(): param.requires_grad = False
		if trainable_blocks == -1:
			for param in self.parameters(): param.requires_grad = True
			print("Model unfrozen: All layers are now trainable.")
			return
		for param in self.head.parameters(): param.requires_grad = True
		if trainable_blocks > 0:
			num_feature_blocks = len(self.features)
			trainable_blocks = min(trainable_blocks, num_feature_blocks)
			for i in range(num_feature_blocks - trainable_blocks, num_feature_blocks):
				for param in self.features[i].parameters(): param.requires_grad = True
		trainable_msg = f"The head and the last {trainable_blocks} feature blocks are trainable." if trainable_blocks > 0 else "Only the head is trainable."
		print(f"Model layers configured: {trainable_msg}")

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
		embedding = torch.flatten(self.avgpool(self.features(x)), 1)
		if self.use_arcface:
			if self.training:
				if labels is None: raise ValueError("Labels are required for ArcFace training.")
				return self.head(embedding, labels)
			else: return F.normalize(embedding, p=2, dim=1)
		else: return self.head(embedding)

	def _fuse_conv_bn(self, block: nn.Sequential):
		fuse_candidates = [[str(i), str(i+1)] for i in range(len(block) - 1) if isinstance(block[i], nn.Conv2d) and isinstance(block[i+1], nn.BatchNorm2d)]
		if fuse_candidates: torch.ao.quantization.fuse_modules(block, fuse_candidates, inplace=True)

	def fuse_model(self):
		if self.training:
			print("⚠️ Warning: Model fusion should be applied in eval mode. No fusion performed.")
			return
		self._fuse_conv_bn(self.features[0])
		for module in self.modules():
			if isinstance(module, (MBConv, FusedMBConv)): self._fuse_conv_bn(module.block)
		print("Model fusion completed successfully.")

# --- Quantization Utilities (from model/quant.py) ---
def prepare_qat_fx(float_model, input_size=(1, 3, 32, 32)):
	qconfig_mapping = quant.get_default_qat_qconfig_mapping('fbgemm')

	example_inputs = torch.rand(size=input_size).cpu()
	prepared_qat = prepare_qat_fx_(float_model, qconfig_mapping, example_inputs=example_inputs)

	return prepared_qat

def prepare_post_static_quantize_fx(float_model, calib_dl, input_size=(1, 3, 32, 32)):
	quant_model = copy.deepcopy(float_model).cpu().eval()
	fuse_model(quant_model)

	qconfig_mapping = quant.get_default_qconfig_mapping("fbgemm")

	example_inputs = torch.rand(size=input_size).cpu()
	prepared = prepare_fx(quant_model, qconfig_mapping, example_inputs=example_inputs)

	# calibration: run a batch through prepared model
	with torch.no_grad():
		for inputs, _ in calib_dl:
			prepared(inputs.cpu())
			break
	
	return prepared

# --- Model Factory (from model/__init__.py) ---
class Model(nn.Module):
	def __init__(self, cfg: Config):
		super().__init__()
		model_params = cfg.model.params
		common_params = {
			"num_classes": model_params.num_classes,
			"pretrained": model_params.pretrained,
			"use_arcface": model_params.use_arcface,
			"s": model_params.s,
			"m": model_params.m,
		}
		if model_params.model_name.startswith("efficientnet_v2"):
			self.model = EfficientNetV2ArcFace(model_name=model_params.model_name, trainable_blocks=model_params.trainable_blocks, **common_params)
		elif model_params.model_name == "panns_cnn14":
			self.model = PannsCNN14ArcFace(**common_params)
		elif model_params.model_name == "panns_mobilenetv1":
			self.model = PannsMobileNetV1ArcFace(**common_params)
		elif model_params.model_name == "panns_mobilenetv2":
			self.model = PannsMobileNetV2ArcFace(**common_params)
		else:
			raise ValueError(f"Model '{model_params.model_name}' not supported.")

		self.use_qat = model_params.use_qat
		if self.use_qat:
			self.model.eval()
			self.model.fuse_model()
			self.model = prepare_qat_fx(self.model)

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)

	def load_state_dict(self, state_dict):
		self.model.load_state_dict(state_dict)

	def state_dict(self):
		return self.model.state_dict()

	def quantise(self, calib_loader=None):
		if not self.use_qat:
			if calib_loader is None:
				raise ValueError("Calibration dataloader is required for post-static quantization.")
			self.model.fuse_model()
			self.model = prepare_post_static_quantize_fx(self.model, calib_loader)
		self.model = convert_fx(self.model)

# --- Dataset (from dataset.py) ---
def index_directory(directory, file_types=('.ogg', '.mp3', '.wav', '.flac')):
	audio_files, labels = [], []
	class_names = sorted([d.name for d in Path(directory).glob('*') if d.is_dir()])
	class_to_idx = {name: i for i, name in enumerate(class_names)}
	for class_name in class_names:
		class_dir = Path(directory) / class_name
		for file_path in class_dir.glob('**/*'):
			if file_path.suffix.lower() in file_types:
				audio_files.append(str(file_path))
				labels.append(class_to_idx[class_name])
	return audio_files, labels, class_names

class SpectrogramDataset(Dataset):
	def __init__(self, audio_files, labels, cfg: Config, audio_transforms=None, image_transforms=None, is_train=False):
		self.audio_files = audio_files
		self.labels = labels
		self.cfg = cfg
		self.audio_transforms = audio_transforms
		self.image_transforms = image_transforms
		self.clip_samples = int(cfg.data.audio_clip_duration * cfg.data.sample_rate)
		self.cache = dc.Cache(cfg.system.cache_directory, size_limit=10**10) if cfg.system.use_disk_cache else None
		self.is_train = is_train

	def _get_cache_key(self, file_path):
		m = hashlib.sha256()
		m.update(str(file_path).encode())
		m.update(str(self.cfg.data.sample_rate).encode())
		# Add other relevant cfg params to hash if they affect the output
		return m.hexdigest()

	def __len__(self):
		return len(self.audio_files)

	def __getitem__(self, idx):
		file_path = self.audio_files[idx]
		label = self.labels[idx]
		cache_key = self._get_cache_key(file_path) if self.cache else None
		if cache_key and cache_key in self.cache:
			spectrogram = self.cache[cache_key]
			return torch.from_numpy(spectrogram).float(), torch.tensor(label).long()

		try:
			audio_data, _ = librosa.load(file_path, sr=self.cfg.data.sample_rate, mono=True)
		except Exception as e:
			print(f"Error loading {file_path}: {e}")
			audio_data = np.zeros(self.clip_samples, dtype=np.float32)

		if len(audio_data) > self.clip_samples:
			start = np.random.randint(0, len(audio_data) - self.clip_samples) if self.is_train else (len(audio_data) - self.clip_samples) // 2
			audio_data = audio_data[start : start + self.clip_samples]
		elif len(audio_data) < self.clip_samples:
			audio_data = np.pad(audio_data, (0, self.clip_samples - len(audio_data)), 'constant')

		if self.audio_transforms:
			audio_data = self.audio_transforms(samples=audio_data, sample_rate=self.cfg.data.sample_rate)

		mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=self.cfg.data.sample_rate, n_fft=self.cfg.data.n_fft, hop_length=self.cfg.data.hop_length, n_mels=self.cfg.data.n_mels, fmin=self.cfg.data.fmin, fmax=self.cfg.data.fmax)
		mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.cfg.data.top_db)
		mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
		spectrogram = np.expand_dims(mel_spec_db, axis=0)
		
		spectrogram_tensor = torch.from_numpy(spectrogram).float()
		if self.image_transforms:
			spectrogram_tensor = self.image_transforms(spectrogram_tensor)
		
		spectrogram = spectrogram_tensor.numpy()
		if cache_key: self.cache[cache_key] = spectrogram
		return torch.from_numpy(spectrogram).float(), torch.tensor(label).long()

# --- Trainer (from train.py) ---
class Trainer:
	def __init__(self, cfg: Config, model, train_loader, val_loader, device, name=None):
		self.cfg = cfg
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		
		self.dtype = getattr(torch, cfg.training.dtype, 'bfloat16')
		self.use_amp = (self.dtype == torch.float16) and ('cuda' in self.device.type)
		self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
		
		self.criterion = self._get_criterion()
		self.optimizer = self._get_optimizer()
		self.scheduler = self._get_scheduler()

		self.name = name if name is not None else "model"
		self.epochs = self.cfg.training.epochs
		self.best_metric = float('inf') if self.cfg.training.metric_mode == 'min' else float('-inf')
		self.metric_mode = self.cfg.training.metric_mode

		timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		self.output_dir = Path(f"outputs/{self.name}/{timestamp}")
		self.output_dir.mkdir(parents=True, exist_ok=True)
		self.best_model_path = self.output_dir / f"best_{self.name}.pth"
		self.writer = SummaryWriter(log_dir=self.output_dir)

	def _get_criterion(self):
		name = self.cfg.training.criterion
		params = vars(self.cfg.training.criterion_params)
		if name == 'CrossEntropyLoss': return nn.CrossEntropyLoss(**params)
		else: raise ValueError(f"Unsupported criterion: {name}")

	def _get_optimizer(self):
		name = self.cfg.training.optimizer
		params = vars(self.cfg.training.optimizer_params)
		if name == 'AdamW': return torch.optim.AdamW(self.model.parameters(), **params)
		else: raise ValueError(f"Unsupported optimizer: {name}")

	def _get_scheduler(self):
		name = self.cfg.training.scheduler
		params = vars(self.cfg.training.scheduler_params)
		if name == 'ReduceLROnPlateau': return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **params)
		else: raise ValueError(f"Unsupported scheduler: {name}")

	def _train_one_epoch(self, pbar, metrics_dict):
		self.model.train()
		running_loss = 0.0
		all_labels, all_predictions = [], []
		for inputs, labels in self.train_loader:
			inputs, labels = inputs.to(self.device), labels.to(self.device)
			self.optimizer.zero_grad()
			with autocast(self.device.type, enabled=self.use_amp):
				outputs = self.model(inputs, labels=labels) if self.cfg.model.params.use_arcface else self.model(inputs)
				loss = self.criterion(outputs, labels)
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			running_loss += loss.item() * inputs.size(0)
			_, predicted = torch.max(outputs.data, 1)
			all_labels.extend(labels.cpu().numpy())
			all_predictions.extend(predicted.cpu().numpy())
			metrics_dict['batch_loss'] = f'{loss.item():.4f}'
			metrics_dict['phase'] = 'train'
			pbar.set_postfix(metrics_dict)
		epoch_loss = running_loss / len(self.train_loader.dataset)
		epoch_acc = accuracy_score(all_labels, all_predictions)
		self.writer.add_scalar('Loss/train', epoch_loss, self.current_epoch)
		self.writer.add_scalar('Accuracy/train', epoch_acc, self.current_epoch)
		return epoch_loss, epoch_acc

	def _evaluate(self, dataloader, pbar, metrics_dict):
		self.model.eval()
		running_loss = 0.0
		all_labels, all_predictions = [], []
		with torch.no_grad():
			for inputs, labels in dataloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				with autocast(self.device.type, enabled=self.use_amp):
					outputs = self.model(inputs, labels=labels) if self.cfg.model.params.use_arcface else self.model(inputs)
					loss = self.criterion(outputs, labels)
				running_loss += loss.item() * inputs.size(0)
				_, predicted = torch.max(outputs.data, 1)
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predicted.cpu().numpy())
				metrics_dict['phase'] = 'val'
				pbar.set_postfix(metrics_dict)
		epoch_loss = running_loss / len(dataloader.dataset)
		epoch_acc = accuracy_score(all_labels, all_predictions)
		return epoch_loss, epoch_acc

	def train(self):
		pbar = tqdm(range(self.epochs), total=self.epochs, desc="Training Progress")
		metrics_dict = {'phase': 'train'}
		for epoch in pbar:
			self.current_epoch = epoch + 1
			train_loss, train_acc = self._train_one_epoch(pbar, metrics_dict)
			val_loss, val_acc = self._evaluate(self.val_loader, pbar, metrics_dict)
			metrics_dict.update({'train_loss': f'{train_loss:.4f}', 'train_acc': f'{train_acc:.4f}', 'val_loss': f'{val_loss:.4f}', 'val_acc': f'{val_acc:.4f}'})
			pbar.set_postfix(metrics_dict)
			self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
			self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)
			current_metric = val_loss if self.cfg.training.metric_mode == 'min' else val_acc
			if (self.metric_mode == 'min' and current_metric < self.best_metric) or (self.metric_mode == 'max' and current_metric > self.best_metric):
				self.best_metric = current_metric
				torch.save(self.model.state_dict(), self.best_model_path)
			if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
				self.scheduler.step(val_loss)
			else:
				self.scheduler.step()
		self.writer.close()
		print(f"Training complete. Best model saved to {self.best_model_path}")

	def test(self, test_loader, model_to_test=None, device=None):
		print("Starting evaluation on test set...")
		eval_device = device if device else self.device
		model = model_to_test.to(eval_device) if model_to_test else self.model.to(eval_device)
		if not model_to_test: model.load_state_dict(torch.load(self.best_model_path, map_location=eval_device))
		model.eval()
		all_labels, all_predictions = [], []
		with torch.no_grad():
			for inputs, labels in tqdm(test_loader, desc="[Testing]"):
				inputs = inputs.to(eval_device)
				with autocast(enabled=self.use_amp):
					# Inference mode for arcface models returns embeddings, which is not what we want for classification test
					if self.cfg.model.params.use_arcface:
						outputs = self.model.model.head.forward(self.model.model(inputs, labels=None), labels)
					else:
						outputs = self.model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predicted.cpu().numpy())
		accuracy = accuracy_score(all_labels, all_predictions)
		precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
		recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
		f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
		print(f"\n--- Test Set Metrics ---\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}\n------------------------")
		return accuracy, precision, recall, f1

# --- Main Execution (from main.py) ---
def main():
	cfg = Config()

	# --- USER CONFIGURATION ---
	# Set your data directory here
	cfg.system.audio_data_directory = 'REPLACE_WITH_YOUR_DATA_DIRECTORY'
	# Override other configs if needed
	# cfg.training.epochs = 10
	# cfg.model.params.model_name = 'panns_cnn14'
	# cfg.model.params.use_arcface = True
	# --- END OF USER CONFIGURATION ---

	if cfg.system.audio_data_directory == 'REPLACE_WITH_YOUR_DATA_DIRECTORY':
		print("Please set `cfg.system.audio_data_directory` to your dataset path.")
		return

	print("--- Configuration ---")
	print(f"Device: {cfg.training.device}")
	print(f"Model: {cfg.model.params.model_name}")
	print(f"Epochs: {cfg.training.epochs}")
	print("---------------------")

	torch.manual_seed(cfg.training.seed)
	device = torch.device(cfg.training.device)
	
	audio_files, labels, class_names = index_directory(cfg.system.audio_data_directory)
	num_classes = len(class_names)
	cfg.data.num_classes = num_classes
	cfg.model.params.num_classes = num_classes

	dataset_size = len(audio_files)
	train_size = int(cfg.data.train_split * dataset_size)
	indices = torch.randperm(dataset_size)
	train_indices = indices[:train_size]
	val_indices = indices[train_size:]

	augs = cfg.augmentations
	train_dataset = SpectrogramDataset([audio_files[i] for i in train_indices], [labels[i] for i in train_indices], cfg, audio_transforms=augs.get_audio_transforms(), image_transforms=augs.get_image_transforms(), is_train=True)
	val_dataset = SpectrogramDataset([audio_files[i] for i in val_indices], [labels[i] for i in val_indices], cfg)
	
	train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

	model = Model(cfg).to(device)
	trainer = Trainer(cfg, model, train_loader, val_loader, device, cfg.model.name)
	trainer.train()

	print("\n--- Testing best floating point model ---")
	trainer.test(val_loader)

	print("\n--- Quantizing model ---")
	trainer.model.load_state_dict(torch.load(trainer.best_model_path, map_location='cpu'))
	quantized_model = trainer.model
	quantized_model.quantise(calib_loader=val_loader)
	
	print("\n--- Testing best quantized model ---")
	trainer.test(val_loader, model_to_test=quantized_model, device='cpu')

if __name__ == "__main__":
	main()
