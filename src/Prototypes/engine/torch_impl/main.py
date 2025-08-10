import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path

from dataset import SpectrogramDataset, index_directory
from models import get_model
from train import train_one_epoch, evaluate

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))

	torch.manual_seed(cfg.training.seed)
	device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
	
	audio_files, labels, class_names = index_directory(cfg.system.audio_data_directory)
	num_classes = len(class_names)
	
	OmegaConf.set_struct(cfg, False)
	cfg.data.num_classes = num_classes
	OmegaConf.set_struct(cfg, True)

	dataset_size = len(audio_files)
	train_size = int(cfg.data.train_split * dataset_size)
	val_size = int(cfg.data.val_split * dataset_size)
	test_size = dataset_size - train_size - val_size
	
	indices = list(range(dataset_size))
	# Note: for full reproducibility you might want to save/load the shuffled indices
	torch.manual_seed(cfg.training.seed)
	torch.random.shuffle(torch.as_tensor(indices))

	train_indices = indices[:train_size]
	val_indices = indices[train_size : train_size + val_size]
	
	train_dataset = SpectrogramDataset(
		[audio_files[i] for i in train_indices],
		[labels[i] for i in train_indices],
		cfg,
		audio_transforms=get_audio_transforms(cfg),
		image_transforms=get_image_transforms(cfg)
	)

	val_dataset = SpectrogramDataset(
		[audio_files[i] for i in val_indices],
		[labels[i] for i in val_indices],
		cfg, audio_transforms=None, image_transforms=None
	)
	
	train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

	model = get_model(cfg).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

	best_val_acc = 0.0
	output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

	for epoch in range(cfg.training.epochs):
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)

		print(f"Epoch {epoch+1}/{cfg.training.epochs}:")
		print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
		print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			torch.save(model.state_dict(), output_dir / "best_model.pth")
			print(f"  New best model saved to {output_dir / 'best_model.pth'}")

if __name__ == "__main__":
	main()
