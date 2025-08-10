import torch
import torch.nn as nn
import torchvision.models as models
from omegaconf import DictConfig

def _create_efficientnet(params: DictConfig):
	model_name = params.model_name
	pretrained = params.pretrained
	num_classes = params.num_classes

	weights = "DEFAULT" if pretrained else None
	model = getattr(models, model_name)(weights=weights)

	# Modify first conv layer for 1-channel input (spectrogram)
	first_conv_layer = model.features[0][0]
	if first_conv_layer.in_channels == 3:
		new_first_conv = nn.Conv2d(
			in_channels=1, 
			out_channels=first_conv_layer.out_channels,
			kernel_size=first_conv_layer.kernel_size,
			stride=first_conv_layer.stride,
			padding=first_conv_layer.padding,
			bias=first_conv_layer.bias is not None,
		)

		if pretrained:
			# Average weights from 3 channels to 1
			original_weights = first_conv_layer.weight.data
			new_weights = original_weights.mean(dim=1, keepdim=True)
			new_first_conv.weight.data = new_weights

		model.features[0][0] = new_first_conv

	# Modify classifier
	in_features = model.classifier[1].in_features
	model.classifier[1] = nn.Linear(in_features, num_classes)

	return model

def get_model(cfg: DictConfig):
	"""
	Builds a model from a hydra config.
	"""
	model_name = cfg.model.name.lower()

	if model_name == 'efficientnetv2':
		return _create_efficientnet(cfg.model.params)
	else:
		raise ValueError(f"Model '{model_name}' not supported.")
