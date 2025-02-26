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
from collections import OrderedDict
import numpy as np

sys.path.append('/homes/marcus/Dokumenter/LDM_fdg/')
from models.resnet_10_mod import resnet10



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
            with_conditioning = True,
            cross_attention_dim=512, 
            )
        
        # Initialize ResNet-10 model for segmentation mask encoding
        self.resnet_10 = resnet10(
            sample_input_D=128,
            sample_input_H=192,
            sample_input_W=192,
            num_seg_classes=119,  # Number of classes (0-117) + 118
            shortcut_type='B',
            no_cuda=False
        )

        # Load the pretrained state dict
        pretrain = torch.load('/homes/marcus/Dokumenter/LDM_fdg/models/resnet_10_23dataset.pth')
        pretrained_dict = pretrain['state_dict']
        new_state_dict = OrderedDict()
        for k, v in pretrained_dict.items():
            if 'conv1' not in k:  # Skip the conv1 weights
                new_state_dict[k] = v
        self.resnet_10.load_state_dict(new_state_dict, strict=False)

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
        return self.model(x, timesteps, context) 


    def training_step(self, batch, batch_idx):
        latents = batch['latents'].float()
        seg = batch['seg'].float()
        
        # Add randomness for CFG: Set 15% of the seg masks to -1
        if torch.rand(1).item() < 0.15: # torch.rand generates a tensor, so we use .item() to get a scalar value
            seg.fill_(118)  # Set all entries in encoded_seg to -1
        encoded_seg = self.resnet_10(seg)
        

        # Add noise to data
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.size(0),), device=self.device).long()
        noise = torch.randn_like(latents)

        # Get model prediction
        noise_pred = self.inferer(inputs=latents, diffusion_model=self.model, noise=noise, timesteps=timesteps, condition=encoded_seg)
        
        # Compute noise loss
        noise_loss = F.mse_loss(noise_pred.float(), noise.float())

        self.log('train_loss', noise_loss, sync_dist=True)
        return noise_loss
    
    def validation_step(self, batch, batch_idx):
        latents = batch['latents'].float()
        seg = batch['seg'].float()
        
        # Add randomness for CFG: Set 15% of the seg masks to -1
        if torch.rand(1).item() < 0.15:  # torch.rand generates a tensor, so we use .item() to get a scalar value
            seg.fill_(118)  # Set all entries in encoded_seg to -1
        encoded_seg = self.resnet_10(seg)

        # Add noise to data
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents.size(0),), device=self.device).long()
        noise = torch.randn_like(latents)

        # Get model prediction
        noise_pred = self.inferer(inputs=latents, diffusion_model=self.model, noise=noise, timesteps=timesteps, condition=encoded_seg)
        
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
                patience=15,        # Number of epochs to wait before reducing LR
                verbose=True,      # Print messages when LR is reduced
                min_lr=1e-9        # Minimum learning rate
            ),
            'monitor': 'val_loss'  # Metric to monitor for plateau
        }
        return [optimizer], [scheduler]