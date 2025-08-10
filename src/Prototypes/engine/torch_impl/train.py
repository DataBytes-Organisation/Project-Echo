import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import hydra
from omegaconf import DictConfig

class Trainer:
	def __init__(self, cfg: DictConfig, model, train_loader, val_loader, device, name=None):
		self.cfg = cfg
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		
		self.criterion = hydra.utils.instantiate(self.cfg.training.criterion)
		self.optimizer = hydra.utils.instantiate(self.cfg.training.optimizer, params=self.model.parameters())
		self.scheduler = hydra.utils.instantiate(self.cfg.training.scheduler, optimizer=self.optimizer)

		self.name = name if name is not None else "model"

		self.epochs = self.cfg.training.epochs
		self.best_metric = float('inf') if self.cfg.training.metric_mode == 'min' else float('-inf')
		self.metric_mode = self.cfg.training.metric_mode
		self.best_model_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / f"best_{self.name}.pth"
		
		self.writer = SummaryWriter(log_dir=self.best_model_path.parent)

	def _train_one_epoch(self, pbar, metrics_dict):
		self.model.train()
		running_loss = 0.0
		all_labels = []
		all_predictions = []
		
		for inputs, labels in self.train_loader:
			inputs, labels = inputs.to(self.device), labels.to(self.device)
			self.optimizer.zero_grad()
			outputs = self.model(inputs)
			loss = self.criterion(outputs, labels)
			loss.backward()
			self.optimizer.step()

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
		all_labels = []
		all_predictions = []

		with torch.no_grad():
			for inputs, labels in dataloader:
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				outputs = self.model(inputs)
				loss = self.criterion(outputs, labels)
				running_loss += loss.item() * inputs.size(0)
				_, predicted = torch.max(outputs.data, 1)
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predicted.cpu().numpy())
				
				metrics_dict['train_loss'] = f'{loss.item():.4f}'
				metrics_dict['phase'] = 'val'
				pbar.set_postfix(metrics_dict)
		
		epoch_loss = running_loss / len(dataloader.dataset)
		epoch_acc = accuracy_score(all_labels, all_predictions)
		
		return epoch_loss, epoch_acc

	def train(self):
		pbar = tqdm(range(self.epochs), total=self.epochs, desc="Training Progress")
		
		metrics_dict = {
			'phase': 'train'
			# 'train_loss': 'N/A',
			# 'train_acc': 'N/A',
			# 'val_loss': 'N/A',
			# 'val_acc': 'N/A',
		}
		
		for epoch in pbar:
			self.current_epoch = epoch + 1
			train_loss, train_acc = self._train_one_epoch(pbar, metrics_dict)
			val_loss, val_acc = self._evaluate(self.val_loader, pbar, metrics_dict)
			
			# metrics_dict['train_loss'] = f'{train_loss:.4f}'
			metrics_dict['train_acc'] = f'{train_acc:.4f}'
			metrics_dict['val_loss'] = f'{val_loss:.4f}'
			metrics_dict['val_acc'] = f'{val_acc:.4f}'
			
			pbar.set_postfix(metrics_dict)

			self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
			self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)

			current_metric = val_loss if self.cfg.training.metric_mode == 'min' else val_acc
			if (self.metric_mode == 'min' and current_metric < self.best_metric) or \
			   (self.metric_mode == 'max' and current_metric > self.best_metric):
				self.best_metric = current_metric
				torch.save(self.model.state_dict(), self.best_model_path)

			# TODO: make this support other scheduler
			self.scheduler.step(val_loss)
				
		self.writer.close()
		print(f"Training complete. Best model saved to {self.best_model_path}")

	def test(self, test_loader, model_to_test=None, device=None):
		print("Starting evaluation on test set...")
		
		eval_device = device if device else self.device
		
		if model_to_test:
			model = model_to_test.to(eval_device)
		else:
			model = self.model.to(eval_device)
			model.load_state_dict(torch.load(self.best_model_path, map_location=eval_device))

		model.eval()

		all_labels = []
		all_predictions = []

		with torch.no_grad():
			for inputs, labels in tqdm(test_loader, desc="[Testing]"):
				inputs, labels = inputs.to(eval_device), labels.to(eval_device)
				outputs = model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				all_labels.extend(labels.cpu().numpy())
				all_predictions.extend(predicted.cpu().numpy())

		accuracy = accuracy_score(all_labels, all_predictions)
		precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
		recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
		f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
		
		print("\n--- Test Set Metrics ---")
		print(f"Accuracy: {accuracy:.4f}")
		print(f"Precision: {precision:.4f}")
		print(f"Recall: {recall:.4f}")
		print(f"F1 Score: {f1:.4f}")
		print("------------------------")
		
		return accuracy, precision, recall, f1

class DistillationTrainer(Trainer):
	def __init__(self, cfg: DictConfig, model, train_loader, val_loader, device):
		super().__init__(cfg, model, train_loader, val_loader, device)

		if not self.cfg.training.distillation.enabled:
			raise ValueError("DistillationTrainer requires distillation to be enabled in config.")

		self.teacher_model = hydra.utils.instantiate(self.cfg.model).to(self.device)
		self.teacher_model.load_state_dict(torch.load(self.cfg.training.distillation.teacher_model_path))
		self.teacher_model.eval()

		self.distillation_criterion = hydra.utils.instantiate(self.cfg.training.distillation.criterion)
		self.alpha = self.cfg.training.distillation.alpha
		self.temperature = self.cfg.training.distillation.temperature

	def _train_one_epoch(self, pbar, metrics_dict):
		self.model.train()
		running_loss = 0.0
		all_labels = []
		all_predictions = []
		
		for inputs, labels in self.train_loader:
			inputs, labels = inputs.to(self.device), labels.to(self.device)
			self.optimizer.zero_grad()

			student_outputs = self.model(inputs)
			
			with torch.no_grad():
				teacher_outputs = self.teacher_model(inputs)

			# Standard cross-entropy loss for the student
			student_loss = self.criterion(student_outputs, labels)

			# TODO: modify it so it also works with CosineEmbedding Loss and MSELoss
			# Distillation loss (KL Divergence between softened student and teacher logits)
			distillation_loss = self.distillation_criterion(
				F.log_softmax(student_outputs / self.temperature, dim=1),
				F.softmax(teacher_outputs / self.temperature, dim=1)
			)

			total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
			
			total_loss.backward()
			self.optimizer.step()

			running_loss += total_loss.item() * inputs.size(0)
			_, predicted = torch.max(student_outputs.data, 1)
			all_labels.extend(labels.cpu().numpy())
			all_predictions.extend(predicted.cpu().numpy())
			
			metrics_dict['batch_loss'] = f'{total_loss.item():.4f}'
			metrics_dict['phase'] = 'train'
			pbar.set_postfix(metrics_dict)
			
		epoch_loss = running_loss / len(self.train_loader.dataset)
		epoch_acc = accuracy_score(all_labels, all_predictions)
		
		self.writer.add_scalar('Loss/train', epoch_loss, self.current_epoch)
		self.writer.add_scalar('Accuracy/train', epoch_acc, self.current_epoch)
		
		return epoch_loss, epoch_acc
