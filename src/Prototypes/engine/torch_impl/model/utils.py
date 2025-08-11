import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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
