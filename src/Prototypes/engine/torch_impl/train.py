import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import hydra
from omegaconf import DictConfig

from model.utils import ArcMarginProduct, CircleLoss
from model import Model
from copy import copy

class Trainer:
	def __init__(self, cfg: DictConfig, model, train_loader, val_loader, device, name=None):
		self.cfg = cfg
		self.model = model.to(device)
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.device = device
		
		self.dtype = getattr(torch, cfg.training.get('dtype', 'bfloat16'))
		self.use_amp = (self.dtype == torch.float16) and ('cuda' in self.device.type)
		self.scaler = GradScaler(self.device.type, enabled=self.use_amp)
		self.grad_accum_steps = self.cfg.training.get('grad_accum_steps', 1)
		
		self.criterion = hydra.utils.instantiate(self.cfg.training.criterion)
		self.optimizer = hydra.utils.instantiate(self.cfg.training.optimizer, params=self.model.parameters())
		self.scheduler = hydra.utils.instantiate(self.cfg.training.scheduler, optimizer=self.optimizer)

		self.grad_clip = self.cfg.training.get('grad_clip', 1.0)

		self.metric_loss_module = None
		use_arcface_cfg = self.cfg.training.get('use_arcface', None)
		if use_arcface_cfg == 'arcface':
			self.metric_loss_module = ArcMarginProduct(
				s=self.cfg.training.arcface.s,
				m=self.cfg.training.arcface.m
			).to(self.device)
		elif use_arcface_cfg == 'circle':
			self.metric_loss_module = CircleLoss(
				s=self.cfg.training.circle.s,
				m=self.cfg.training.circle.m
			).to(self.device)

		self.distillation = self.cfg.training.distillation.enabled
		self.teacher_model_path = self.cfg.training.distillation.teacher_model_path
		self.distillation_alpha = self.cfg.training.distillation.alpha
		self.distillation_temperature = self.cfg.training.distillation.temperature

		if self.distillation:
			try:
				_model = copy(self.cfg.model)
				self.cfg.model = copy(self.cfg.teacher_model)

				self.teacher_model = Model(self.cfg)
				self.teacher_model.load_state_dict(torch.load(self.teacher_model_path, map_location=torch.device("cpu")))

				self.teacher_model = self.teacher_model.to(self.device)
				self.teacher_model.eval()

				self.cfg.model = _model
				print("Succesfully loaded teacher model.")

				self.distillation_criterion = hydra.utils.instantiate(self.cfg.training.distillation.criterion)
			except Exception as e:
				print("Failed to load teacher model. Revert back to normal training.")
				print(e)
				self.cfg.training.distillation.enabled = False

		self.name = name if name is not None else "model"

		self.epochs = self.cfg.training.epochs
		self.best_metric = float('inf') if self.cfg.training.metric_mode == 'min' else float('-inf')
		self.metric_mode = self.cfg.training.metric_mode
		self.best_model_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / f"best_{self.name}.pth"

		self.early_stopping_patience = self.cfg.training.get('early_stopping_patience', 3)
		self.epochs_no_improve = 0
		self.best_epoch = 0
		
		self.writer = SummaryWriter(log_dir=self.best_model_path.parent)

	def save_checkpoint(self, path, epoch, best=True):
		"""Save training checkpoint."""
		checkpoint = {
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
			'best_metric': self.best_metric,
			'epochs_no_improve': self.epochs_no_improve,
			'best_epoch': self.best_epoch,
			'cfg': OmegaConf.to_container(self.cfg, resolve=True),
		}

		torch.save(checkpoint, path)
		print(f"Checkpoint saved to {path}")

		if best:
			torch.save(checkpoint, self.best_model_path)
			print(f"Best model checkpoint saved to {self.best_model_path}")
	
	def load_checkpoint(self, path):
		"""Load training checkpoint to resume training."""

		if path.is_file():
			checkpoint = torch.load(path, map_location=self.device)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
				self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
			
			self.start_epoch = checkpoint.get('epoch', 0)
			self.best_metric = checkpoint.get('best_metric', self.best_metric)
			self.epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
			self.best_epoch = checkpoint.get('best_epoch', 0)
			
			print(f"Resumed from epoch {self.start_epoch}, best metric {self.best_metric:.4f}")
		else:
			raise FileNotFoundError(f"Checkpoint not found at {path}")

	def _train_one_epoch(self, pbar, metrics_dict):
		self.model.train()
		running_loss = 0.0
		all_labels = []
		all_predictions = []
		
		for batch_idx, (inputs, labels) in enumerate(self.train_loader):
			inputs, labels = inputs.to(self.device), labels.to(self.device)
			self.optimizer.zero_grad()
			
			with autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
				student_outputs = self.model(inputs)
				if self.metric_loss_module is not None:
					student_outputs = self.metric_loss_module(student_outputs, labels)

				if self.distillation and self.teacher_model:
					with torch.no_grad():
						teacher_outputs = self.teacher_model(inputs)

					student_loss = self.criterion(student_outputs, labels)
					distillation_loss = self.distillation_criterion(
						F.log_softmax(student_outputs / self.distillation_temperature, dim=1),
						F.softmax(teacher_outputs / self.distillation_temperature, dim=1)
					)
					loss = (1. - self.distillation_alpha) * student_loss + self.distillation_alpha * distillation_loss * (self.distillation_temperature ** 2)
				else:
					loss = self.criterion(student_outputs, labels)

				loss = loss / self.grad_accum_steps

			self.scaler.scale(loss).backward()

			# if self.grad_clip is not None:
			# 	# Unscale the gradients before clipping to avoid clipping scaled gradients
			# 	self.scaler.unscale_(self.optimizer)
			# 	torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

			if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
				if self.grad_clip is not None:
					self.scaler.unscale_(self.optimizer)
					torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

				self.scaler.step(self.optimizer)
				self.scaler.update()
				
				# Flush gradients after the update
				self.optimizer.zero_grad()

			# self.scaler.step(self.optimizer)
			# self.scaler.update()

			running_loss += (loss.item() * self.grad_accum_steps) * inputs.size(0)

			_, predicted = torch.max(student_outputs.data, 1)
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
				with autocast(self.device.type, enabled=self.use_amp):
					outputs = self.model(inputs)
					if self.metric_loss_module is not None:
						outputs = self.metric_loss_module(outputs, labels)

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
		
		metrics_dict = {
			'phase': 'train',
			'best_metric': f'{self.best_metric:.4f}'
		}
		
		for epoch in pbar:
			self.current_epoch = epoch + 1
			train_loss, train_acc = self._train_one_epoch(pbar, metrics_dict)
			val_loss, val_acc = self._evaluate(self.val_loader, pbar, metrics_dict)
			
			metrics_dict['train_loss'] = f'{train_loss:.4f}'
			metrics_dict['train_acc'] = f'{train_acc:.4f}'
			metrics_dict['val_loss'] = f'{val_loss:.4f}'
			metrics_dict['val_acc'] = f'{val_acc:.4f}'
			
			self.writer.add_scalar('Loss/val', val_loss, self.current_epoch)
			self.writer.add_scalar('Accuracy/val', val_acc, self.current_epoch)

			current_metric = val_loss if self.cfg.training.metric_mode == 'min' else val_acc
			
			# Check for improvement
			is_better = (self.metric_mode == 'min' and current_metric < self.best_metric) or \
						(self.metric_mode == 'max' and current_metric > self.best_metric)
			
			if is_better:
				self.best_metric = current_metric
				self.best_epoch = self.current_epoch
				torch.save(self.model.state_dict(), self.best_model_path)
				self.epochs_no_improve = 0
				metrics_dict['best_metric'] = f'{self.best_metric:.4f}'
			else:
				self.epochs_no_improve += 1

			pbar.set_postfix(metrics_dict)

			if self.scheduler:
				if isinstance(self.scheduler, ReduceLROnPlateau):
					self.scheduler.step(val_loss)
				else:
					self.scheduler.step()
			
			# Check for early stopping
			if self.epochs_no_improve >= self.early_stopping_patience:
				print(f"\nEarly stopping triggered after {self.epochs_no_improve} epochs with no improvement.")
				break
				
		self.writer.close()
		print(f"\nTraining complete. Best model from epoch {self.best_epoch} saved to {self.best_model_path}")

		print("Loading best model weights...")
		self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
		print("Best model loaded successfully.")

	def test(self, test_loader, model_to_test=None, device=None):
		print("Starting evaluation on test set...")
		
		eval_device = device if device else self.device
		
		if model_to_test:
			model = model_to_test.to(eval_device)
		else:
			model = self.model.to(eval_device)

		model.eval()

		all_labels = []
		all_predictions = []

		with torch.no_grad():
			for inputs, labels in tqdm(test_loader, desc="[Testing]"):
				inputs, labels = inputs.to(self.device), labels.to(self.device)
				with autocast(self.device.type, enabled=self.use_amp):
					outputs = self.model(inputs)

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
