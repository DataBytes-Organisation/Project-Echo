import torch
import torch.nn as nn
import torchvision.models as models

from omegaconf import DictConfig

from model.effv2 import EfficientNetV2ArcFace
# from model.panns_cnn14 import PannsCNN14ArcFace
# from model.panns_mobilenetv1 import PannsMobileNetV1ArcFace
# from model.panns_mobilenetv2 import PannsMobileNetV2ArcFace
from model.quant import prepare_qat_fx, prepare_post_static_quantize_fx, convert_fx

class Model(nn.Module):
	def __init__(self, cfg: DictConfig):
		super().__init__()
		model_params = cfg.model.params

		if model_params.model_name.startswith("efficientnet_v2"):
			self.model = EfficientNetV2ArcFace(
				model_name=model_params.model_name,
				pretrained=model_params.pretrained,
				num_classes=model_params.num_classes,
				use_arcface=model_params.get("use_arcface", False),
				s=model_params.get("s", 30.0),
				m=model_params.get("m", 0.50),
				trainable_blocks=model_params.get("trainable_blocks", 0),
			)
		# elif model_params.model_name == "panns_cnn14":
		# 	self.model = PannsCNN14ArcFace(
		# 		classes_num=model_params.classes_num,
		# 		pretrained=model_params.get("pretrained", True),
		# 		use_arcface=model_params.get("use_arcface", False),
		# 		s=model_params.get("s", 30.0),
		# 		m=model_params.get("m", 0.50)
		# 	)
		# elif model_params.model_name == "panns_mobilenetv1":
		# 	self.model = PannsMobileNetV1ArcFace(
		# 		classes_num=model_params.classes_num,
		# 		pretrained=model_params.get("pretrained", True),
		# 		use_arcface=model_params.get("use_arcface", False),
		# 		s=model_params.get("s", 30.0),
		# 		m=model_params.get("m", 0.50)
		# 	)
		# elif model_params.model_name == "panns_mobilenetv2":
		# 	self.model = PannsMobileNetV2ArcFace(
		# 		num_classes=model_params.num_classes,
		# 		pretrained=model_params.get("pretrained", True),
		# 		use_arcface=model_params.get("use_arcface", False),
		# 		s=model_params.get("s", 30.0),
		# 		m=model_params.get("m", 0.50)
		# 	)
		else:
			raise ValueError(f"Model '{model_params.model_name}' not supported.")

		self.use_qat = model_params.get("use_qat", False)
		if self.use_qat:
			# self.model = fuse_model(self.model)
			self.model.eval()
			self.model.fuse_model() 
			self.model = prepare_qat_fx(self.model)
	
	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)

	def load_state_dict(self, state_dict):
		self.model.load_state_dict(state_dict)

	def state_dict(self):
		return self.model.state_dict()

	def quantise(self):
		if not self.use_qat:
			self.model = fuse_model(self.model)
			self.model = prepare_post_static_quantize_fx(self.model)

		self.model = convert_fx(self.model)
