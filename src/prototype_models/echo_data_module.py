import pytorch_lightning as pl
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split, sampler

from htsat_utils import do_mixup, get_mix_lambda, do_mixup_label
from htsat_utils import get_loss_func, d_prime, float32_to_int16

import baseline_config

class EchoDataModule(pl.LightningDataModule):
    
    def __init__(self, train_dataset, eval_dataset, device_num):
        super().__init__()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device_num = device_num

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle = False) if self.device_num > 1 else None
        train_loader = DataLoader(
            dataset = self.train_dataset,
            num_workers = baseline_config.num_workers,
            batch_size = baseline_config.batch_size // self.device_num,
            shuffle = False,
            sampler = train_sampler
        )
        return train_loader
    
    def val_dataloader(self):
        eval_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        eval_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = baseline_config.num_workers,
            batch_size = baseline_config.batch_size // self.device_num,
            shuffle = False,
            sampler = eval_sampler
        )
        return eval_loader
    
    def test_dataloader(self):
        test_sampler = DistributedSampler(self.eval_dataset, shuffle = False) if self.device_num > 1 else None
        test_loader = DataLoader(
            dataset = self.eval_dataset,
            num_workers = baseline_config.num_workers,
            batch_size = baseline_config.batch_size // self.device_num,
            shuffle = False,
            sampler = test_sampler
        )
        
        return test_loader