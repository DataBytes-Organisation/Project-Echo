import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.ao.quantization as quant
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, prepare_qat_fx as prepare_qat_fx_

import numpy as np

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
