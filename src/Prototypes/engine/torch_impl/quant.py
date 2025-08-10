import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# TODO: make it within custom model since each model 
# would have different arhcitecture and require 
# different type of fusing. 
#
# By doing that it would make the quantised version
# smaller and more efficient
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

def prepare_qat_fx(float_model, input_size=(1, 3, 32, 32)):
	qconfig = quant.get_default_qat_qconfig('fbgemm')
	qconfig_dict = {"": qconfig}

	example_inputs = torch.rand(size=input_size).cpu()
	prepared_qat = prepare_qat_fx_(float_model, qconfig_dict, example_inputs=example_inputs)

	return prepared_qat

