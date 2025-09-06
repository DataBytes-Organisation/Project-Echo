import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quant 
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx as prepare_qat_fx_

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

import time
import numpy as np
import pandas as pd
import random

INPUT_SIZE = 224
transforms_train = transforms.Compose([
	transforms.Resize(INPUT_SIZE),
  transforms.RandomCrop(32, padding=4),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transforms_test = transforms.Compose([
	transforms.Resize(INPUT_SIZE),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_train)
test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

class LightNN(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()

		self.features = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(16, 16, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)

		self.classifier = nn.Sequential(
			nn.Linear(1024, 256),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(256, num_classes)
		)

	def get_features_map(self, x):
		x = self.features(x)
		return x

	def get_emb(self, x):
		features = self.get_features_map(x)
		x = torch.flatten(features, 1)
		return features, x

	def classify(self, emb):
		x = self.classifier(emb)
		return x

	def forward(self, x):
		_, x = self.get_emb(x)
		x = self.classify(x)
		return x

def fuse_model(model):
	"""
	Fuse conv+relu and linear+relu in simple Sequential modules.
	In-place on the given model.
	"""
	fused_model = copy.deepcopy(model).cpu().eval()

	modules_to_fuse = []

	# features is a Sequential
	try:
		features = model.features
		for i in range(len(features) - 1):
			if isinstance(features[i], nn.Conv2d) and isinstance(features[i+1], nn.ReLU):
				modules_to_fuse.append([f"features.{i}", f"features.{i+1}"])
	except Exception:
		pass

	# classifier sequential (linear + relu)
	try:
		classifier = model.classifier
		for i in range(len(classifier) - 1):
			if isinstance(classifier[i], nn.Linear) and isinstance(classifier[i+1], nn.ReLU):
				modules_to_fuse.append([f"classifier.{i}", f"classifier.{i+1}"])
	except Exception:
		pass

	if modules_to_fuse:
		quant.fuse_modules(fused_model, modules_to_fuse, inplace=True)

	return fused_model

def prepare_example_input_from_loader(dl, device):
	batch = next(iter(dl))[0]
	example_input = batch[:1].cpu()

	return (example_input,)

def prepare_post_static_quantize_fx(float_model, calib_dl, input_size=(1, 3, 32, 32)):
	quant_model = copy.deepcopy(float_model).cpu().eval()
	fuse_model(quant_model)

	qconfig = quant.get_default_qconfig("fbgemm")
	qconfig_dict = {"": qconfig}

	example_inputs = torch.rand(size=input_size).cpu()
	prepared = prepare_fx(quant_model, qconfig_dict, example_inputs=example_inputs)

	# calibration: run a batch through prepared model
	with torch.no_grad():
		for inputs, _ in calib_dl:
			prepared(inputs.cpu())
			break
	
	return prepared

# From: https://gist.github.com/TadaoYamaoka/9db512cfd504d66c114263565eb2fbde#file-qat_fx-py-L139
def measure_inference_latency(model, device, input_size=(1, 3, 32, 32), num_samples=100, num_warmups=10):
	model.to(device)
	model.eval()

	x = torch.rand(size=input_size).to(device)

	with torch.no_grad():
		for _ in range(num_warmups):
			_ = model(x)

	if torch.cuda.is_available():
		torch.cuda.synchronize()

	with torch.no_grad():
		start_time = time.time()
		for _ in range(num_samples):
			_ = model(x)
			if torch.cuda.is_available():
				torch.cuda.synchronize()
		end_time = time.time()

	elapsed_time = end_time - start_time
	elapsed_time_ave = elapsed_time / num_samples

	return elapsed_time_ave

# Based on: https://gist.github.com/TadaoYamaoka/9db512cfd504d66c114263565eb2fbde#file-qat_fx-py-L192
def model_equivalence(model_1, model_2, device, rtol=1e-05, atol=1e-08, num_tests=100, input_size=(1, 3, 32, 32)):

	model_1.to(device)
	model_2.to(device)

	model_1.eval()
	model_2.eval()

	for _ in range(num_tests):
		x = torch.rand(size=input_size).to(device)

		y1 = model_1(x).detach().cpu().numpy()
		y2 = model_2(x).detach().cpu().numpy()

		if np.allclose(a=y1, b=y2, rtol=rtol, atol=atol, equal_nan=False) == False:
			print("Model equivalence test sample failed: ")

			print(y1)
			print(y2)

			return False

	return True

def get_model_size(model):
  """Get model size in MB"""
  param_size = 0
  for param in model.parameters():
    param_size += param.nelement() * param.element_size()

  buffer_size = 0
  for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / (1024 ** 2)

  return size_all_mb

def get_memory_size(model):
	"""Calculates the in-memory size of a PyTorch model."""
	total_size = 0

	for param in model.parameters():
		total_size += param.nelement() * param.element_size()

	for buffer in model.buffers():
		total_size += buffer.nelement() * buffer.element_size()

	return total_size / (1024 * 1024)

def get_memory_size(model):
	"""
	Calculates the in-memory size of a PyTorch model, including quantized models.
	Returns size in MB.
	"""
	total_size = 0
	
	for name, param in model.named_parameters():
		if param is not None:
			if hasattr(param, 'dtype'):
				if param.dtype == torch.qint8:
					# 1 byte per qint8
					param_size = param.numel() * 1

					# Add scale and zero_point overhead (typically float32)
					# 4 bytes each for scale and zero_point
					param_size += 8
				elif param.dtype == torch.quint8:
					param_size = param.numel() * 1
					param_size += 8
				elif param.dtype == torch.qint32:
					# 4 bytes per qint32
					param_size = param.numel() * 4
					param_size += 8
				else:
					param_size = param.numel() * param.element_size()
			else:
				param_size = param.numel() * param.element_size()
			
			total_size += param_size
	
	# Handle buffers (including quantized buffers)
	for name, buffer in model.named_buffers():
		if buffer is not None:
			if hasattr(buffer, 'dtype'):
				if buffer.dtype == torch.qint8:
					buffer_size = buffer.numel() * 1
					buffer_size += 8  # scale and zero_point
				elif buffer.dtype == torch.quint8:
					buffer_size = buffer.numel() * 1
					buffer_size += 8
				elif buffer.dtype == torch.qint32:
					buffer_size = buffer.numel() * 4
					buffer_size += 8
				else:
					buffer_size = buffer.numel() * buffer.element_size()
			else:
				buffer_size = buffer.numel() * buffer.element_size()
			
			total_size += buffer_size
	
	total_size += _get_quantized_module_overhead(model)
	
	return total_size / (1024 * 1024)

def _get_quantized_module_overhead(model):
	"""Calculate additional overhead from quantized modules"""
	overhead = 0
	
	for module in model.modules():
		if hasattr(module, '_packed_params'):
			# Approximate overhead for packed params
			overhead += 64
		
		# Check for quantized conv layers
		if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
			if module.weight.dtype in [torch.qint8, torch.quint8, torch.qint32]:
				# Additional overhead for quantized convolutions
				overhead += 32
		
		# Check for fake quantization modules (used during QAT)
		if 'FakeQuantize' in str(type(module)):
			# Overhead for fake quantization observers
			overhead += 16
	
	return overhead

def prepare_qat_fx(float_model, input_size=(1, 3, 32, 32)):
	qconfig = quant.get_default_qat_qconfig('fbgemm')
	qconfig_dict = {"": qconfig}

	example_inputs = torch.rand(size=input_size).cpu()
	prepared_qat = prepare_qat_fx_(model, qconfig_dict, example_inputs=example_inputs)

	return prepared_qat

device = torch.device('cpu')

RANDOM_SEED = 0

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
  torch.cuda.manual_seed(RANDOM_SEED)
  torch.cuda.manual_seed_all(RANDOM_SEED)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False # Disable the non-deterministic algorithms

effnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
in_features = effnet.classifier[1].in_features
effnet.classifier[1] = nn.Linear(in_features, 10)
print(effnet)

# light_model = LightNN().to(device)

fused_effnet = fuse_model(effnet)
quantised_effnet = prepare_post_static_quantize_fx(fused_effnet, test_dl)
quantised_effnet = convert_fx(quantised_effnet)
quantised_effnet_jit = torch.jit.script(quantised_effnet)

print(fused_effnet)
print(model_equivalence(effnet, fused_effnet, device))

print(quantised_effnet)
print(quantised_effnet_jit)

fp32_cpu_inference_latency = measure_inference_latency(model=fused_effnet, device=device, num_samples=100)
int8_cpu_inference_latency = measure_inference_latency(model=quantised_effnet, device=device, num_samples=100)
int8_jit_cpu_inference_latency = measure_inference_latency(model=quantised_effnet_jit, device=device, num_samples=100)

print("FP32 CPU Inference Latency: {:.2f} ms / sample".format(fp32_cpu_inference_latency * 1000))
print("INT8 CPU Inference Latency: {:.2f} ms / sample".format(int8_cpu_inference_latency * 1000))
print("INT8 JIT CPU Inference Latency: {:.2f} ms / sample".format(int8_jit_cpu_inference_latency * 1000))

torch.save(fused_effnet.state_dict(), "fp32_effnet.pth")
torch.save(quantised_effnet.state_dict(), "int8_effnet.pth")
torch.jit.save(quantised_effnet_jit, "int8_jit_effnet.pth")

fp32_size = os.path.getsize("fp32_effnet.pth") / (1024 * 1024)
int8_size = os.path.getsize("int8_effnet.pth") / (1024 * 1024)
int8_jit_size = os.path.getsize("int8_jit_effnet.pth") / (1024 * 1024)

print(f"FP32 Model Size:       {fp32_size:.2f} MB")
print(f"INT8 Eager Model Size: {int8_size:.2f} MB  (Saved as FP32 by state_dict)")
print(f"INT8 JIT Model Size:   {int8_jit_size:.2f} MB  (True INT8 deployment size)")

fp32_mem_size = get_memory_size(fused_effnet)
int8_mem_size = get_memory_size(quantised_effnet)
int8_jit_mem_size = get_memory_size(quantised_effnet_jit)

# --- Final Comparison ---
print("\n--- Model Size Comparison ---")
print("                          | FP32 Model | INT8 Eager | INT8 JIT")
print("--------------------------|------------|------------|-----------")
print(f"On-Disk Size (MB)         | {fp32_size:10.2f} | {int8_size:10.2f} | {int8_jit_size:9.2f}")
print(f"In-Memory Size (MB)       | {fp32_mem_size:10.2f} | {int8_mem_size:10.2f} | {int8_jit_mem_size:9.2f}")
