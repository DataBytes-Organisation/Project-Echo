import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import copy
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from dataset import SpectrogramDataset, index_directory, validation_collate_fn
from model import Model
from train import Trainer



@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
	print(OmegaConf.to_yaml(cfg))

	torch.manual_seed(cfg.training.seed)
	device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
	
	audio_files, labels, class_names = index_directory(cfg.system.audio_data_directory)
	num_classes = len(class_names)

	# Save the class names to the output directory for later use
	try:
		output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
		class_names_path = output_dir / "class_names.txt"
		with open(class_names_path, "w") as f:
			for name in class_names:
				f.write(f"{name}\n")
		print(f"Saved class names to {class_names_path}")
	except Exception as e:
		print(f"Warning: Could not save class names file: {e}")
	
	OmegaConf.set_struct(cfg, False)
	cfg.data.num_classes = num_classes
	OmegaConf.set_struct(cfg, True)

	dataset_size = len(audio_files)
	train_size = int(cfg.data.train_split * dataset_size)
	val_size = int(cfg.data.val_split * dataset_size)
	# test_size = dataset_size - train_size - val_size
	
	# indices = torch.tensor(list(range(dataset_size)))
	# torch.manual_seed(cfg.training.seed)
	# shuffled_indices = torch.randperm(len(indices))
	# indices = indices[shuffled_indices]
	# # torch.random.shuffle(torch.as_tensor(indices))
	#
	# train_indices = indices[:train_size]
	# # val_indices = indices[train_size : train_size + val_size]
	# val_indices = indices[train_size:]
	# # test_indices = indices[train_size + val_size:]
	
	print(f"Splitting dataset with stratification (Val split: {cfg.data.val_split})...")
	class_indices = defaultdict(list)
	for idx, label in enumerate(labels):
		class_indices[label].append(idx)

	train_indices = []
	val_indices = []

	# Iterate over each class and split independently
	for label, indices in class_indices.items():
		indices = torch.tensor(indices)
		
		# Shuffle indices for this specific class
		# (uses the same seed set at start of main)
		shuffled_class_indices = indices[torch.randperm(len(indices))]
		
		# Calculate split sizes for this class
		n_total = len(indices)
		n_val = int(n_total * cfg.data.val_split)
		
		# Determine split points
		# If a class is too small (e.g. 1 sample), n_val might be 0. 
		# You might want to force at least 1 val sample if n_total > 1, 
		# but strict percentage is standard.
		
		val_subset = shuffled_class_indices[:n_val]
		train_subset = shuffled_class_indices[n_val:]
		
		val_indices.extend(val_subset.tolist())
		train_indices.extend(train_subset.tolist())

	print(f"Total Training samples: {len(train_indices)}")
	print(f"Total Validation samples: {len(val_indices)}")

	audio_transforms = hydra.utils.instantiate(cfg.augmentations.audio) if 'augmentations' in cfg and 'audio' in cfg.augmentations else None
	image_transforms = hydra.utils.instantiate(cfg.augmentations.image) if 'augmentations' in cfg and 'image' in cfg.augmentations else None
	
	train_dataset = SpectrogramDataset(
		[audio_files[i] for i in train_indices],
		[labels[i] for i in train_indices],
		cfg,
		audio_transforms=audio_transforms,
		image_transforms=image_transforms,
		is_train=True,
	)

	val_dataset = SpectrogramDataset(
		[audio_files[i] for i in val_indices],
		[labels[i] for i in val_indices],
		cfg, audio_transforms=None, image_transforms=None
	)
	
	# test_dataset = SpectrogramDataset(
	# 	[audio_files[i] for i in test_indices],
	# 	[labels[i] for i in test_indices],
	# 	cfg, audio_transforms=None, image_transforms=None
	# )
	
	train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, pin_memory=True)
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=validation_collate_fn, pin_memory=True)
	# test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, pin_memory=True)

	# cfg.training.epochs = 15

	model = Model(cfg).to(device)
	print(model.summary())
	trainer = Trainer(cfg, model, train_loader, val_loader, device, cfg.model.name)
	# trainer.train()
	# trainer.test(val_loader)
	# trainer.model.quantise()
	# trainer.test(val_loader)

	if cfg.run.train:
		trainer.train()
	elif cfg.run.test:
		load_path = None

		if cfg.run.checkpoint_path:
			ckpt_path = Path(hydra.utils.to_absolute_path(cfg.run.cKLDivLossheckpoint_path))
			if ckpt_path.is_file():
				load_path = ckpt_path
			else:
				print(f"ERROR: Specified checkpoint '{ckpt_path}' not found!")
		elif trainer.best_model_path.is_file():
			load_path = trainer.best_model_path

		if load_path:
			print(f"Training skipped. Loading model from: {load_path}")
			trainer.model.load_state_dict(torch.load(load_path, map_location=device))
		else:
			print("WARNING: Training skipped and no valid model checkpoint found. Testing with an untrained model.")

	if cfg.run.test:
		print("\n--- Testing original model ---")
		# trainer.test(val_loader, name="Original")
		trainer.test(val_loader)

		if cfg.run.quantise:
			print("\n--- Quantising model ---")
			trainer.model.quantise()
			print("\n--- Testing quantised model ---")
			# trainer.test(val_loader, name="Quantised")
			trainer.test(val_loader)

	# # --- Non-QAT EfficientNetV2-S ---
	# print("--- Training non-QAT EfficientNetV2-S ---")
	# cfg.model.name = 'efficientnetv2'
	# cfg.model.params.model_name = 'efficientnet_v2_s'
	# model_s = get_model(cfg).to(device)
	# trainer_s = Trainer(cfg, model_s, train_loader, val_loader, device, "EfficientNetV2-S")
	# trainer_s.train()
	# trainer_s.test(test_loader)
	# print("-" * 20)

	# # --- QAT EfficientNetV2-S ---
	# print("\n--- Training QAT EfficientNetV2-S ---")
	# cfg.model.name = 'efficientnetv2'
	# cfg.model.params.model_name = 'efficientnet_v2_s'
	# model_s_qat_base = get_model(cfg)
	# model_s_qat_base.to('cpu')
	# fused_model = fuse_model(model_s_qat_base)
	# qat_model = prepare_qat_fx(fused_model)
	
	# trainer_qat = Trainer(cfg, qat_model, train_loader, val_loader, device, "EfficientNetV2-S-QAT")
	# trainer_qat.train()
	
	# # Manually test after converting
	# qat_model.load_state_dict(torch.load(trainer_qat.best_model_path))
	# quantized_model = convert_fx(qat_model.to('cpu'))
	# trainer_qat.test(test_loader, model_to_test=quantized_model, device='cpu')
	# print("-" * 20)
	
	# --- Non-QAT EfficientNetV2-M ---
	# print("\n--- Training non-QAT EfficientNetV2-M ---")
	# cfg.model.name = 'efficientnetv2'
	# cfg.model.params.model_name = 'efficientnet_v2_m'
	# model_m = get_model(cfg).to(device)
	# trainer_m = Trainer(cfg, model_m, train_loader, val_loader, device)
	# trainer_m.train()
	# trainer_m.test(test_loader)
	# print("-" * 20)

if __name__ == "__main__":
	main()
