#!/usr/bin/env python3

import torch
import torchio as tio
from torch.utils.data import DataLoader
import sys
import os
import json
from monai.networks.schedulers import DDIMScheduler
import numpy as np
from tqdm import tqdm
from collections.abc import Callable
import h5py
import nibabel as nib
import gc
from pathlib import Path

# Add paths for helper and model modules
project_root = '/homes/marcus/Dokumenter/LDM_fdg'
sys.path.extend([project_root])

# Import necessary modules and helper functions
from helpers.preprocessing import create_subjects_dataset, create_transforms_AdaIN
from helpers.ldm_helpers import adaptive_instance_normalization
from models.modelvqvae_perc_adv import VQVAE3D
from CFG.ldm_pinaya_CFG import DiffusionModelLightning
from inferer_CFG import DiffusionInferer


class DeterministicInference:
    def __init__(self, model, finetuned_vqvae, h5_file_path):
        self.model = model
        self.finetuned_vqvae = finetuned_vqvae
        self.h5_file_path = h5_file_path
        self.scheduler = DDIMScheduler(
                                    num_train_timesteps=1000,
                                    schedule="scaled_linear_beta",  # Default schedule
                                    beta_start=0.0015,                # Default beta start
                                    beta_end=0.0195                   # Default beta end
                                    )  # Fixed total timesteps
        self.style_file = self._load_h5_data(h5_file_path)
        self.guidance = 50

    def forward_diffusion(self, latents, label_data, style_mean, style_variance, forward_timestep, num_forward_steps):
        """
        Iteratively adds noise to latents using the learned noise model.
        
        Args:
            latents: The input latent representation.
            forward_timestep: The maximum noise timestep to reach.
            num_forward_steps: The number of steps to reach the maximum noise level.
        
        Returns:
            Noisy latents at the specified forward_timestep.
        """
        latents = adaptive_instance_normalization(latents, style_mean, style_variance)

        # Move alphas_cumprod to the same device as latents
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(latents.device)

        # Compute the timesteps to iterate over
        timesteps = torch.linspace(0, forward_timestep, num_forward_steps + 1, dtype=torch.long, device=latents.device)

        noisy_latents = latents.clone()
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]

            # Ensure t is a 1D tensor with correct batch size
            t = t.view(-1).expand(latents.size(0))
             
            # Get conditional prediction
            encoded_cond = self.model.resnet_10(label_data)
            noise_pred_cond = self.model(noisy_latents, t, encoded_cond)

            # Get unconditional prediction
            label_data.fill_(118)
            encoded_uncond = self.model.resnet_10(label_data)
            noise_pred_uncond = self.model(noisy_latents, t, encoded_uncond)

            # Combined noise prediction
            final_pred = noise_pred_uncond + self.guidance * (noise_pred_uncond - noise_pred_cond)

            # Compute scaling factors for t
            alpha_t = self.scheduler.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            # Estimate the clean latent using the first equation
            estimated_clean_latent = (1.0 / sqrt_alpha_t) * (noisy_latents - sqrt_one_minus_alpha_t * final_pred)

            # Compute scaling factors for t+1
            alpha_t_next = self.scheduler.alphas_cumprod[t_next].view(-1, 1, 1, 1, 1)
            sqrt_alpha_t_next = torch.sqrt(alpha_t_next)
            sqrt_one_minus_alpha_t_next = torch.sqrt(1 - alpha_t_next)

            # Compute the next noisy latent using the second equation
            noisy_latents = sqrt_alpha_t_next * estimated_clean_latent + sqrt_one_minus_alpha_t_next * final_pred

        return noisy_latents

    def create_custom_timesteps(self, forward_timestep: int, num_reverse_timestep: int) -> torch.Tensor:
        step_ratio = forward_timestep // num_reverse_timestep  # Calculate the decrement step
        timesteps = torch.arange(forward_timestep, -1, -step_ratio, dtype=torch.long)  # Generate the sequence
        timesteps = timesteps[1:num_reverse_timestep]  # Remove the first entry and ensure correct number of steps
        timesteps = torch.cat([timesteps, torch.tensor([0], dtype=torch.long)])  # Add 0 at the end
        return timesteps
    
    # run_inference remains the same
    def run_inference(self, content_image, label_data, forward_timestep=30, num_forward_steps=5, num_reverse_steps=6): #guidance in inferer = 15
        with torch.no_grad():
            content_latents = self.finetuned_vqvae.encoder(content_image)
            
            style_means, style_variances = [], []
            for idx in range(content_latents.size(0)):
                unique_labels = torch.unique(label_data[idx][label_data[idx] > 0]).cpu().numpy()
                label_key = str(tuple(sorted(unique_labels)))
                style_mean, style_variance = self.get_style_statistics(label_key)
                style_means.append(style_mean.to(content_latents.device))
                style_variances.append(style_variance.to(content_latents.device))

            style_mean = torch.stack(style_means, dim=0)
            style_variance = torch.stack(style_variances, dim=0)

            # Add noise up to forward_timestep
            noisy_latents = self.forward_diffusion(content_latents, label_data, style_mean, style_variance, forward_timestep, num_forward_steps)

            # Set reverse steps and and define inferer
            self.scheduler.set_timesteps(num_reverse_steps)
            costum_timesteps = self.create_custom_timesteps(forward_timestep, num_reverse_steps)
            self.scheduler.timesteps = costum_timesteps
            inferer = DiffusionInferer(self.scheduler, self.guidance)

            # Get conditional prediction
            pred = inferer.sample(input_noise=noisy_latents, diffusion_model=self.model, scheduler=self.scheduler, conditioning=label_data)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise ValueError("Detected NaNs or Infs in the latents before decoding!")

            # Decode
            _, quantized_latents, _ = self.finetuned_vqvae.vector_quantizer(pred)
            harmonized_image = self.finetuned_vqvae.decoder(quantized_latents)
            return harmonized_image
    
    def _load_h5_data(self, h5_file_path):
        """Load HDF5 data and cache it as a dictionary."""
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



def reconstruct_image_with_style_transfer(diffusion_model, finetuned_vqvae, subject, device, h5_file_path):
    patch_size = (192, 192, 128)
    patch_overlap = (96, 96, 64)  # Adjusted overlap
    crop_dim = (192 // 2, 192 // 2, 128 // 2)

    # Initialize GridSampler and DataLoader
    grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=1)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')

    # Initialize inference class
    inference = DeterministicInference(model=diffusion_model, finetuned_vqvae=finetuned_vqvae, h5_file_path=h5_file_path)

    with torch.no_grad():
        for patches_batch in patch_loader:
            input_tensor = patches_batch['pet'][tio.DATA].to(device).float()
            label_data = patches_batch['label'][tio.DATA].to(device)

            reconstructed_patches = inference.run_inference(input_tensor, label_data)

            # Aggregate reconstructed patches
            aggregator.add_batch(reconstructed_patches.cpu(), patches_batch[tio.LOCATION])
            torch.cuda.empty_cache() 

    # Reassemble the reconstructed image
    reconstructed_image = aggregator.get_output_tensor()

    # Crop to original dimensions
    cropped_image = reconstructed_image[
        :, crop_dim[0]:-crop_dim[0],
        crop_dim[1]:-crop_dim[1],
        crop_dim[2]:-crop_dim[2]
    ]
    return cropped_image.squeeze(0)

## LOAD MODELS ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
finetuned_vqvae_checkpoint_path = "/depict/users/oscar/shared/V3_H2O_FineTune-epoch=59-val_loss=0.02156120.ckpt"
ldm_checkpoint_path = "/depict/users/marcus/shared/LDM_fdg_checkpoints/ldm_fdg_pinaya_CFG_it2/fdg_LDM_pinaya_it2-periodic-epoch=120.ckpt"
h5_file_path = "/homes/marcus/Dokumenter/LDM_precomputed_latens/AdaIN_computations/data/all_patients_V3H2O_agg.h5"


# Load the pre-trained VQ-VAE
finetuned_vqvae = VQVAE3D(
    in_channels=1,
    latent_dim=4,
    num_embeddings=512,
    learning_rate=0.0001)
vqvae_checkpoint = torch.load(finetuned_vqvae_checkpoint_path)
# Filter out 'resnet' keys from the checkpoint
filtered_state_dict = {k: v for k, v in vqvae_checkpoint['state_dict'].items() if not k.startswith('resnet')}
finetuned_vqvae.load_state_dict(filtered_state_dict, strict=False)
# Set the model to evaluation mode
finetuned_vqvae.to(device)
finetuned_vqvae.eval()
for param in finetuned_vqvae.parameters():
    param.requires_grad = False

# Initialize DiffusionModelUNet
diffusion_model = DiffusionModelLightning(
                                        in_channels=4, 
                                        out_channels=4, 
                                        h5_file_path=h5_file_path)

# Initialize LatentDiffusionModel with VQVAE and Diffusion Model
diffusion_model_checkpoint = torch.load(ldm_checkpoint_path)
diffusion_model.load_state_dict(diffusion_model_checkpoint['state_dict'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
diffusion_model.to(device)
diffusion_model.eval()
for param in diffusion_model.parameters():
    param.requires_grad = False


#val_path = '/homes/marcus/Dokumenter/splits/h2o_val_list.json'
val_path = '/depict/users/marcus/shared/data_splits_h2o/h2o_test_list.json'
with open(val_path, 'r') as file:
    val_list = json.load(file)

patch_size3d_val = (192, 192, 128)
_, validation_transforms = create_transforms_AdaIN(patch_size3d_val, patch_size3d_val)
dataset_val = create_subjects_dataset(val_list, transforms=validation_transforms)

# Output paths
output_base_path = "/depict/users/marcus/shared/style_transfered_images/fdg_pinaya_CFG_e120_30_6_5_st_testset"
output_json_path = "/depict/users/marcus/shared/style_transfered_images/fdg_pinaya_CFG_e120_30_6_5_st_testset.json"  # Update with the desired output path

# Prepare the output JSON structure
output_subject_list = []

for subject in dataset_val:
    # Extract original paths
    pet_path = subject['pet'].path
    seg_path = subject['label'].path

    # Derive the relative paths to maintain the folder structure in the output
    pet_relative_path = Path(pet_path).relative_to("/depict/users/marcus/shared/h2o_q99_intnorm")
    seg_relative_path = Path(seg_path).relative_to("/depict/users/marcus/shared/h2o_q99_intnorm")

    # Update the output paths for PET and segmentation
    pet_output_path = Path(output_base_path) / pet_relative_path
    seg_output_path = Path(output_base_path) / seg_relative_path.parent.parent / "seg_orig" / seg_relative_path.name


    # Ensure output directories exist
    pet_output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the PET image
    pet_img = nib.load(pet_path)

    # Load the segmentation mask
    seg_img = nib.load(seg_path)

    # Perform style transfer reconstruction
    reconstructed_image = reconstruct_image_with_style_transfer(
        diffusion_model, finetuned_vqvae, subject, device, h5_file_path
    )

    # Save the reconstructed PET image
    nib.save(
        nib.Nifti1Image(reconstructed_image.cpu().numpy(), pet_img.affine),
        pet_output_path
    )
    print(f"Reconstructed PET saved to: {pet_output_path}")

    # Copy the segmentation mask to the new location
    nib.save(seg_img, seg_output_path)
    print(f"Segmentation mask copied to: {seg_output_path}")

    # Append the new paths to the output JSON structure
    output_subject_list.append({
        "pet": str(pet_output_path),
        "seg_orig": str(seg_output_path)
    })

# Save the new JSON file
with open(output_json_path, 'w') as file:
    json.dump(output_subject_list, file, indent=4)

print(f"New JSON file saved to: {output_json_path}")
