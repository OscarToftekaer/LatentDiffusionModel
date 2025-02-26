#!/usr/bin/env python3

import pytorch_lightning as pl
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
import torch
import torch.nn.functional as F
import sys
# Add the path to the helpers folder
sys.path.append('/homes/marcus/Dokumenter/LDM_fdg/helpers/')
from monai.networks.schedulers import DDPMScheduler
from monai.inferers import DiffusionInferer



class DiffusionModelLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels, h5_file_path,lr=1e-4):
        super().__init__()
        self.model = DiffusionModelUNet(
            spatial_dims=3,  # 3D model
            in_channels=in_channels,
            out_channels=out_channels,
            num_res_blocks=(2, 2, 2),  # Can be tuned
            num_channels=(256, 512, 768), # Can be tuned
            attention_levels=(False, True, True),
            num_head_channels=(0, 512, 768),  # Head channels for self-attention
            resblock_updown=False,  # No up/downsampling within residual blocks
            norm_num_groups=32,  # Normalization groups
            norm_eps=1e-10,  # Use attention in deeper layers
            )

        self.lr = lr
        self.h5_file_path = h5_file_path
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            schedule="scaled_linear_beta",
            beta_start=0.0015,
            beta_end=0.0195,
            )
        self.inferer = DiffusionInferer(self.scheduler)


    def forward(self, x, timesteps, context=None):
        return self.model(x, timesteps) 


    def training_step(self, batch, batch_idx):
        latents = batch['latents'].float()

        # Add noise to data
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.size(0),), device=self.device).long()
        noise = torch.randn_like(latents)

        # Get model prediction
        noise_pred = self.inferer(inputs=latents, diffusion_model=self.model, noise=noise, timesteps=timesteps)
        
        # Compute noise loss
        noise_loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log('train_loss', noise_loss, sync_dist=True)
        return noise_loss
    
    def validation_step(self, batch, batch_idx):
        latents = batch['latents'].float()

        # Add noise to data
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.size(0),), device=self.device).long()
        noise = torch.randn_like(latents)

        # Get model prediction
        noise_pred = self.inferer(inputs=latents, diffusion_model=self.model, noise=noise, timesteps=timesteps)
        
        # Compute noise loss
        val_loss = F.mse_loss(noise_pred.float(), noise.float())
        self.log('val_loss', val_loss, sync_dist=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',        # Minimize the monitored metric
                factor=0.1,        # Reduce the learning rate by a factor of 0.1
                patience=10,        # Number of epochs to wait before reducing LR
                verbose=True,      # Print messages when LR is reduced
                min_lr=1e-9        # Minimum learning rate
            ),
            'monitor': 'val_loss'  # Metric to monitor for plateau
        }
        return [optimizer], [scheduler]