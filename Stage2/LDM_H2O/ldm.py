#!/usr/bin/env python3

import pytorch_lightning as pl
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
import torch
import torch.nn.functional as F
import h5py
import sys
# Add the path to the helpers folder
sys.path.append('/homes/marcus/Dokumenter/LDM_precomputed_latens/helpers/')
from ldm_helpers import adaptive_instance_normalization
from monai.networks.schedulers import DDPMScheduler
from inferer import DiffusionInferer



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

        # Load the style statistics from the HDF5 file
        self.style_file = self._load_h5_data(h5_file_path)

    def forward(self, x, timesteps, context=None):
        return self.model(x, timesteps)
    
    def _load_h5_data(self, h5_file_path):
        """Load HDF5 data and save it as a dictionary."""
        style_file = {}
        with h5py.File(h5_file_path, 'r') as h5f:
            for label_key in h5f.keys():
                aggregated_mean = h5f[label_key]['aggregated_mean'][:]
                aggregated_variance = h5f[label_key]['aggregated_variance'][:]
                style_file[label_key] = {
                    'mean': torch.tensor(aggregated_mean, dtype=torch.float32),
                    'variance': torch.tensor(aggregated_variance, dtype=torch.float32),
                }

            # Add fallback for global "all" statistics
            if 'all' in h5f:
                style_file['all'] = {
                    'mean': torch.tensor(h5f['all']['aggregated_mean'][:], dtype=torch.float32),
                    'variance': torch.tensor(h5f['all']['aggregated_variance'][:], dtype=torch.float32),
                }
        return style_file

    def get_style_statistics(self, label_key):
        """Retrieve mean and variance from the style file."""
        if label_key in self.style_file:
            return self.style_file[label_key]['mean'], self.style_file[label_key]['variance']
        return self.style_file['all']['mean'], self.style_file['all']['variance']
    
    def compute_style_loss(self, recon_latents, style_mean, style_variance):
        # Ensure style_mean and style_variance are PyTorch tensors
        if not isinstance(style_mean, torch.Tensor):
            style_mean = torch.tensor(style_mean, device=recon_latents.device)
        if not isinstance(style_variance, torch.Tensor):
            style_variance = torch.tensor(style_variance, device=recon_latents.device)

        # Convert variance to standard deviation
        style_std = torch.sqrt(style_variance + 1e-10)

        # Reshape style_mean and style_std to match latent dimensions
        # style_mean, style_std: [batch_size, num_channels] -> [batch_size, num_channels, 1, 1, 1]
        style_mean = style_mean.view(style_mean.size(0), style_mean.size(1), 1, 1, 1)
        style_std = style_std.view(style_std.size(0), style_std.size(1), 1, 1, 1)

        # Compute channel-wise mean and standard deviation of the reconstructed latents
        latents_mean = recon_latents.mean(dim=(2, 3, 4), keepdim=True)  # Shape: [batch_size, num_channels, 1, 1, 1]
        latents_std = recon_latents.std(dim=(2, 3, 4), keepdim=True, unbiased=False)  # Same shape as latents_mean

        # Style loss as the sum of squared differences between latents statistics and target statistics
        mean_loss = F.mse_loss(latents_mean, style_mean)
        std_loss = F.mse_loss(latents_std, style_std)
        style_loss = mean_loss + std_loss

        return style_loss
    
    def instance_normalize(self, tensor):
        mean = tensor.mean(dim=(2, 3, 4), keepdim=True)
        std = tensor.std(dim=(2, 3, 4), keepdim=True) + 1e-10  # Add epsilon to prevent division by zero
        return (tensor - mean) / std


    def training_step(self, batch, batch_idx):
        latents = batch['latents'].float()
        label_comb = batch['labels']

        ## Adaptive Instance Normalization ##
        # Initialize lists to hold style means and variances
        style_means = []
        style_variances = []

        for i in range(latents.size(0)):  # Loop over each sample in the batch
            label_key = str(tuple(sorted(label_comb[i])))  # Convert to string format for HDF5 lookup

            # Retrieve AdaIN mean and variance from cache
            style_mean, style_variance = self.get_style_statistics(label_key)
            style_means.append(style_mean.to(latents.device))
            style_variances.append(style_variance.to(latents.device))

        # Stack style means and variances to create batch tensors
        style_mean = torch.stack(style_means, dim=0).to(latents.device)  # Shape: [batch_size, num_channels]
        style_variance = torch.stack(style_variances, dim=0).to(latents.device)  # Shape: [batch_size, num_channels]

        # Apply AdaIN to the latent representation
        latents_adain = adaptive_instance_normalization(latents, style_mean, style_variance)

        # Add noise to data
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents_adain.size(0),), device=self.device).long()
        noise = torch.randn_like(latents_adain)

        # Get model prediction
        noise_pred, noisy_latents = self.inferer(inputs=latents_adain, diffusion_model=self.model, noise=noise, timesteps=timesteps)
        
        # Compute noise loss
        noise_loss = F.mse_loss(noise_pred.float(), noise.float())

        # Reconstruct latents at t=0 using Equation (6) from Wu et al.
        alpha_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)  
        recon_latents = (1.0 / torch.sqrt(alpha_t)) * (noisy_latents - torch.sqrt(1 - alpha_t) * noise_pred)

        # Compute style loss
        style_loss = self.compute_style_loss(recon_latents, style_mean, style_variance)

        # Compute content loss
        latents_IN = self.instance_normalize(latents)
        recon_latents_IN = self.instance_normalize(recon_latents)
        content_loss = F.mse_loss(recon_latents_IN, latents_IN)

        # Combined loss
        lamba_nl = 1
        lambda_sl = 1  # set to 0.1 in Wu et al.
        lambda_cl = 0.0001
        loss = lamba_nl * noise_loss + lambda_sl * style_loss + lambda_cl * content_loss
        self.log('train_loss', loss, sync_dist=True)
        self.log('noise_loss', lamba_nl*noise_loss, sync_dist=True)
        self.log('style_loss', lambda_sl*style_loss, sync_dist=True)
        self.log('content_loss', lambda_cl*content_loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        latents = batch['latents'].float()
        label_comb = batch['labels']

        ## Adaptive Instance Normalization ##
        style_means = []
        style_variances = []

        for i in range(latents.size(0)):  # Loop over each sample
            label_key = str(tuple(sorted(label_comb[i])))
            style_mean, style_variance = self.get_style_statistics(label_key)
            style_means.append(style_mean.to(latents.device))
            style_variances.append(style_variance.to(latents.device))

        # Stack style means and variances
        style_mean = torch.stack(style_means, dim=0).to(latents.device)
        style_variance = torch.stack(style_variances, dim=0).to(latents.device)

        # Apply AdaIN
        latents_adain = adaptive_instance_normalization(latents, style_mean, style_variance)

        # Add noise
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (latents_adain.size(0),), device=self.device).long()
        noise = torch.randn_like(latents_adain)

        # Use inferer to get noise_pred and noisy_latents, just like in training
        with torch.no_grad():
            noise_pred, noisy_latents = self.inferer(
                inputs=latents_adain,
                diffusion_model=self.model,
                noise=noise,
                timesteps=timesteps
            )

        # Compute noise loss
        noise_loss = F.mse_loss(noise_pred, noise)

        # Reconstruct latents at t=0
        alpha_t = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
        recon_latents = (1.0 / torch.sqrt(alpha_t)) * (noisy_latents - torch.sqrt(1 - alpha_t) * noise_pred)

        # Compute style loss
        style_loss = self.compute_style_loss(recon_latents, style_mean, style_variance)

        # Compute content loss
        latents_IN = self.instance_normalize(latents)
        recon_latents_IN = self.instance_normalize(recon_latents)
        content_loss = F.mse_loss(recon_latents_IN, latents_IN)

        # Combine losses
        lamba_nl = 1
        lambda_sl = 1  # set to 0.1 in Wu et al.
        lambda_cl = 0.0001
        loss = lamba_nl * noise_loss + lambda_sl * style_loss + lambda_cl * content_loss

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_noise_loss', lamba_nl * noise_loss, sync_dist=True)
        self.log('val_style_loss', lambda_sl * style_loss, sync_dist=True)
        self.log('val_content_loss', lambda_cl * content_loss, sync_dist=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',        # Minimize the monitored metric
                factor=0.1,        # Reduce the learning rate by a factor of 0.1
                patience=50,       # Number of epochs to wait before reducing LR
                verbose=True,      # Print messages when LR is reduced
                min_lr=1e-9        # Minimum learning rate
            ),
            'monitor': 'val_loss'  # Metric to monitor for plateau
        }
        return [optimizer], [scheduler]


