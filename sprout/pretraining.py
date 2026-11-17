"""Pixel-space epsilon-prediction pretraining, adapted from DPS's LDM trainer."""

from pathlib import Path
import random

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

from sprout.utils import make_beta_schedule, extract_into_tensor


class PretrainingImages(Dataset):
    """One image path per line; relative paths are resolved against data_root."""

    def __init__(self, image_list, data_root, size=256, augment=True):
        self.paths = [Path(data_root) / line.strip()
                      for line in Path(image_list).read_text(encoding='utf-8').splitlines()
                      if line.strip()]
        if not self.paths:
            raise ValueError('The training image list is empty.')
        missing = next((path for path in self.paths if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(missing)
        if size < 32 or size % 32:
            raise ValueError('Image size must be a positive multiple of 32.')
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise ValueError(f'Cannot read image: {self.paths[index]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # A contained rotated square avoids the original loader's black borders
        # and silent fallback when its crop coordinates fall outside the image.
        if self.augment and random.random() < 0.5:
            h, w = image.shape[:2]
            angle = random.uniform(-90, 90)
            radians = np.deg2rad(angle)
            side = max(1, int(min(h, w) * random.uniform(0.5, 1.0)
                              / (abs(np.cos(radians)) + abs(np.sin(radians)))))
            matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h))
            y, x = (h - side) // 2, (w - side) // 2
            image = image[y:y + side, x:x + side]
        image = cv2.resize(image, (self.size, self.size))
        if self.augment and random.random() < 0.5:
            image = cv2.flip(image, 0)  # DPS uses a vertical flip.
        image = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(image.transpose(2, 0, 1).copy())


class ModelEMA(nn.Module):
    """DPS/LDM's adaptive EMA decay and checkpoint buffer names."""

    def __init__(self, model):
        super().__init__()
        self.register_buffer('decay', torch.tensor(0.9999))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))
        self.names = {}
        for name, parameter in model.named_parameters():
            shadow = name.replace('.', '')
            if shadow in self.names.values():
                raise ValueError(f'EMA parameter name collision: {name}')
            self.names[name] = shadow
            self.register_buffer(shadow, parameter.detach().clone())

    @torch.no_grad()
    def forward(self, model):
        self.num_updates.add_(1)
        decay = min(self.decay.item(), (1 + self.num_updates.item())
                    / (10 + self.num_updates.item()))
        for name, parameter in model.named_parameters():
            getattr(self, self.names[name]).lerp_(parameter.detach(), 1 - decay)


class DiffusionPretraining(pl.LightningModule):
    def __init__(self, backbone, learning_rate=3.5e-5, warmup_steps=10000,
                 timesteps=1000, linear_start=0.0002, linear_end=0.02):
        super().__init__()
        if learning_rate <= 0 or warmup_steps < 0 or timesteps <= 0:
            raise ValueError('Invalid learning rate, warmup steps, or timesteps.')
        if not 0 < linear_start <= linear_end < 1:
            raise ValueError('Diffusion betas must satisfy 0 < start <= end < 1.')
        # Preserve the key prefix consumed by all existing downstream scripts.
        self.model = nn.Module()
        self.model.diffusion_model = backbone
        self.model_ema = ModelEMA(self.model)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.timesteps = timesteps
        betas = torch.tensor(make_beta_schedule('linear', timesteps,
                             linear_start, linear_end), dtype=torch.float64)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        self.register_buffer('betas', betas.float())
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt().float())
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             (1 - alphas_cumprod).sqrt().float())
        self.register_buffer('scale_factor', torch.tensor(1.0))
        self.register_buffer('scale_initialized', torch.tensor(False))

    @torch.no_grad()
    def initialize_scale(self, images):
        if self.scale_initialized.item():
            return
        std = images.float().std()
        # All ranks validate the same value before entering the training loop.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(std, src=0)
        if not torch.isfinite(std) or std <= 0:
            raise ValueError('The first training batch must have nonzero, finite variance.')
        self.scale_factor.copy_(std.reciprocal())
        self.scale_initialized.fill_(True)

    def q_sample(self, images, timesteps, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, timesteps, images.shape) * images
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                      timesteps, images.shape) * noise)

    def forward(self, images, timesteps):
        return self.model.diffusion_model(images, timesteps)

    def training_step(self, batch, batch_idx):
        self.initialize_scale(batch)
        images = batch * self.scale_factor
        timesteps = torch.randint(self.timesteps, (len(images),), device=images.device)
        noise = torch.randn_like(images)
        prediction = self(self.q_sample(images, timesteps, noise), timesteps)
        loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
        self.log('train/loss_simple', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('scale_factor', self.scale_factor, on_step=True, on_epoch=False)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.model_ema(self.model)

    def on_save_checkpoint(self, checkpoint):
        # Lightning 1.4 adds one even when saving after fit() has returned.
        # This scheduler advances exactly once per completed optimizer step.
        if 'lr_schedulers' in checkpoint:
            checkpoint['global_step'] = checkpoint['lr_schedulers'][0]['last_epoch']

    def lr_multiplier(self, step):
        if self.warmup_steps and step < self.warmup_steps:
            return 1e-6 + (1 - 1e-6) * step / self.warmup_steps
        return 1.0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_multiplier)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
