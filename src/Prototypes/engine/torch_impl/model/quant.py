import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.ao.quantization as quant
import numpy as np

import torch
import torch.ao.quantization as quantization
from torch.ao.quantization.quantize_fx import (
	prepare_qat_fx as __prepare_qat_fx,
	prepare_fx,
	convert_fx,
)

default_qat_qconfig = quantization.get_default_qat_qconfig("fbgemm")

per_tensor_weight_handler = quantization.FakeQuantize.with_args(
	observer=quantization.MovingAverageMinMaxObserver,  # Per-tensor observer
	quant_min=-128,
	quant_max=127,
	dtype=torch.qint8,
	qscheme=torch.per_tensor_affine,
	reduce_range=False,
)

qconfig_per_tensor = quantization.QConfig(activation=default_qat_qconfig.activation, weight=per_tensor_weight_handler)

qconfig_mapping = (
	quantization.QConfigMapping()
	.set_object_type(torch.nn.Conv2d, qconfig_per_tensor)
	.set_object_type(torch.nn.Linear, qconfig_per_tensor)
	.set_object_type(torch.nn.BatchNorm2d, qconfig_per_tensor)
	.set_object_type(torch.nn.ReLU, qconfig_per_tensor)
	.set_object_type(torch.nn.ReLU6, qconfig_per_tensor)
)


def prepare_qat_fx(float_model, input_size=(1, 3, 32, 32)):
	example_inputs = torch.rand(size=input_size).cpu()
	prepared_qat = __prepare_qat_fx(float_model, qconfig_mapping, example_inputs=example_inputs)

	return prepared_qat


def prepare_post_static_quantize_fx(float_model, calib_dl, input_size=(1, 3, 32, 32)):
	quant_model = copy.deepcopy(float_model).cpu().eval()

	example_inputs = torch.rand(size=input_size).cpu()
	prepared = prepare_fx(quant_model, qconfig_mapping, example_inputs=example_inputs)

	# calibration: run a batch through prepared model
	with torch.no_grad():
		for inputs, _ in calib_dl:
			prepared(inputs.cpu())
			break

	return prepared
