import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import umap
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class CosineLinear(nn.Module):
	def __init__(self, in_features, out_features, bias=False):
		super().__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

		if bias:
			self.bias = nn.Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.kaiming_uniform_(self.weight, a=5)

		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, input):
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		if self.bias is not None:
			cosine += self.bias

		return cosine

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class ArcMarginProduct(nn.Module):
	def __init__(self, s: float = 30.0, m: float = 0.50):
		"""
		ArcFace loss to make the classification of each class more distinc
		Based on https://arxiv.org/abs/1801.07698
		"""
		super().__init__()

		self.s = s
		self.m = m

		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - self.m) * self.m

	def forward(self, cosine: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
		sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

		phi = cosine * self.cos_m - sine * self.sin_m
		phi = torch.where(cosine > self.th, phi, cosine - self.mm)

		one_hot = torch.zeros(cosine.size(), device=cosine.device)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)

		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
		output = output *  self.s

		return output

class CircleLoss(nn.Module):
	def __init__(self, s=80, m=0.4):
		"""
		Based on https://ieeexplore.ieee.org/document/9156774
		"""
		super().__init__()

		self.s = s
		self.m = m
		
	def forward(self, cosine: torch.Tensor, label: torch.Tensor = None) -> torch.Tensor:
		one_hot = torch.zeros_like(cosine)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		
		# Calculate margins and scaling factors
		alpha_p = F.relu(-cosine.detach() + 1 + self.m)
		alpha_n = F.relu(cosine.detach() + self.m)
		delta_p = 1 - self.m
		delta_n = self.m

		# Calculate logits for positive and negative pairs
		s_p = self.s * (cosine - delta_p)
		s_n = self.s * (cosine - delta_n)
		
		pred_class_logits = one_hot * s_p + (1.0 - one_hot) * s_n
		
		return pred_class_logits

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
			outputs = model(inputs) # return embedding
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
		random_state=0,
		metric='cosine'
	)
	embedding_2d = reducer.fit_transform(embeddings)

	plt.figure(figsize=(14, 12))
	unique_labels = np.unique(labels)
	colors = plt.cm.get_cmap('jet', len(unique_labels))

	for i, label_idx in enumerate(unique_labels):
		class_indices = np.where(labels == label_idx)[0]
		plt.scatter(
			embedding_2d[class_indices, 0],
			embedding_2d[class_indices, 1],
			color=colors(i),
			label=class_names[label_idx],
			alpha=0.7,
			s=10
		)

	plt.title(title, fontsize=16)
	plt.xlabel('UMAP Dimension 1', fontsize=12)
	plt.ylabel('UMAP Dimension 2', fontsize=12)
	plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, fontsize=10)
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.tight_layout(rect=[0, 0, 0.85, 1])
	
	if save_path:
		plt.savefig(save_path, bbox_inches='tight', dpi=300)
		print(f"Plot saved to {save_path}")
	
	plt.show()
