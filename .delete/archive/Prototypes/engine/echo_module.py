import warnings
warnings.filterwarnings("ignore")

# some basic libraries
import sys  
import os
import seaborn as sn
import numpy as np
from numpy.lib.function_base import average
import math
import bisect
import pickle
import random
from platform import python_version


# pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# pytorch
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.optim as optim
import torch.distributed as dist

from torch.nn.parameter import Parameter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

from torchmetrics import Accuracy
from torchvision import transforms
from torchlibrosa.stft import STFT, ISTFT, magphase

# librosa audio processing
import librosa

# sound file
import soundfile as sf

# sklearn machine learning library
from sklearn import metrics
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn import preprocessing

from htsat_utils import do_mixup, get_mix_lambda, do_mixup_label
from htsat_utils import get_loss_func, d_prime, float32_to_int16

class EchoModule(pl.LightningModule):
    def __init__(self, sed_model, config, dataset):
        super().__init__()
        self.sed_model = sed_model
        self.config = config
        self.dataset = dataset
        self.loss_func = get_loss_func(config.loss_type)

    def evaluate_metric(self, pred, ans):
        acc = accuracy_score(ans, np.argmax(pred, 1))
        return {"acc": acc}  
    
    def forward(self, x, mix_lambda = None):
        output_dict = self.sed_model(x, mix_lambda)
        return output_dict["clipwise_output"], output_dict["framewise_output"]

    def inference(self, x):
        self.device_type = next(self.parameters()).device
        self.eval()
        x = torch.from_numpy(x).float().to(self.device_type)
        output_dict = self.sed_model(x, None, True)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        return output_dict

    def training_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        mix_lambda = None
        pred, _ = self(batch["melspec"], mix_lambda)
        loss = self.loss_func(pred, batch["target"])
        self.log("loss", loss, on_epoch= True, prog_bar=True)
        return loss
        
    def training_epoch_end(self, outputs):
        print("")
        # re-shuffle the audio dataset
        # self.dataset.generate_queue()

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch["melspec"])
        return [pred.detach(), batch["target"].detach()]
    
    def validation_epoch_end(self, validation_step_outputs):
        self.device_type = next(self.parameters()).device
        pred = torch.cat([d[0] for d in validation_step_outputs], dim = 0)
        target = torch.cat([d[1] for d in validation_step_outputs], dim = 0)

        if torch.cuda.device_count() > 1:
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            dist.barrier()

        metric_dict = {
            "acc":0.
        }
        
        if torch.cuda.device_count() > 1:
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                if self.config.dataset_type == "scv2":
                    gather_target = np.argmax(gather_target, 1)
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
        
            self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier()
        else:
            gather_pred = pred.cpu().numpy()
            gather_target = target.cpu().numpy()
            metric_dict = self.evaluate_metric(gather_pred, gather_target)
            # print(self.device_type, metric_dict, flush = True)  
            self.log("acc", metric_dict["acc"], on_epoch = True, prog_bar=True, sync_dist=False)
            
        
    def time_shifting(self, x, shift_len):
        shift_len = int(shift_len)
        new_sample = torch.cat([x[:, shift_len:], x[:, :shift_len]], axis = 1)
        return new_sample 

    def test_step(self, batch, batch_idx):
        self.device_type = next(self.parameters()).device
        preds = []
        # time shifting optimization
        if self.config.fl_local or self.config.dataset_type != "audioset": 
            shift_num = 1 # framewise localization cannot allow the time shifting
        else:
            shift_num = 10 
        for i in range(shift_num):
            pred, pred_map = self(batch["melspec"])
            preds.append(pred.unsqueeze(0))
            batch["melspec"] = self.time_shifting(batch["melspec"], shift_len = 100 * (i + 1))
        preds = torch.cat(preds, dim=0)
        pred = preds.mean(dim = 0)
        if self.config.fl_local:
            return [
                pred.detach().cpu().numpy(), 
                pred_map.detach().cpu().numpy(),
                batch["audio_name"],
                batch["real_len"].cpu().numpy()
            ]
        else:
            return [pred.detach(), batch["target"].detach()]

    def test_epoch_end(self, test_step_outputs):
        self.device_type = next(self.parameters()).device
        if self.config.fl_local:
            pred = np.concatenate([d[0] for d in test_step_outputs], axis = 0)
            pred_map = np.concatenate([d[1] for d in test_step_outputs], axis = 0)
            audio_name = np.concatenate([d[2] for d in test_step_outputs], axis = 0)
            real_len = np.concatenate([d[3] for d in test_step_outputs], axis = 0)
            heatmap_file = os.path.join(self.config.heatmap_dir, self.config.test_file + "_" + str(self.device_type) + ".npy")
            save_npy = [
                {
                    "audio_name": audio_name[i],
                    "heatmap": pred_map[i],
                    "pred": pred[i],
                    "real_len":real_len[i]
                }
                for i in range(len(pred))
            ]
            np.save(heatmap_file, save_npy)
        else:
            self.device_type = next(self.parameters()).device
            pred = torch.cat([d[0] for d in test_step_outputs], dim = 0)
            target = torch.cat([d[1] for d in test_step_outputs], dim = 0)
            gather_pred = [torch.zeros_like(pred) for _ in range(dist.get_world_size())]
            gather_target = [torch.zeros_like(target) for _ in range(dist.get_world_size())]
            dist.barrier()
            metric_dict = {
                "acc":0.
            }
            dist.all_gather(gather_pred, pred)
            dist.all_gather(gather_target, target)
            if dist.get_rank() == 0:
                gather_pred = torch.cat(gather_pred, dim = 0).cpu().numpy()
                gather_target = torch.cat(gather_target, dim = 0).cpu().numpy()
                if self.config.dataset_type == "scv2":
                    gather_target = np.argmax(gather_target, 1)
                metric_dict = self.evaluate_metric(gather_pred, gather_target)
                print(self.device_type, dist.get_world_size(), metric_dict, flush = True)
            self.log("acc", metric_dict["acc"] * float(dist.get_world_size()), on_epoch = True, prog_bar=True, sync_dist=True)
            dist.barrier() 

    def configure_optimizers(self):
        
        # network optimiser
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr = self.config.learning_rate, 
            betas = (0.9, 0.999), eps = 1e-08, weight_decay = self.config.htsat_weight_decay, 
        )
        
        # learning rate scheduler
        def lr_foo(epoch):       
            if epoch < 3:
                # warm up lr
                lr_scale = self.config.lr_rate[epoch]
            else:
                # warmup schedule
                lr_pos = int(-1 - bisect.bisect_left(self.config.lr_scheduler_epoch, epoch))
                if lr_pos < -3:
                    lr_scale = max(self.config.lr_rate[0] * (0.98 ** epoch), 0.03 )
                else:
                    lr_scale = self.config.lr_rate[lr_pos]
            return lr_scale

        # construct the learning rate scheduler
        #scheduler = optim.lr_scheduler.LambdaLR(
        #    optimizer,
        #    lr_lambda=lr_foo
        #)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                  mode='max',
                                                  patience=20,
                                                  factor=0.80,
                                                  verbose=True)
        
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "acc",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        
        
        return { "optimizer": optimizer, "lr_scheduler":lr_scheduler_config }