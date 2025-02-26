#!/usr/bin/env python3

import os
import sys
import random
import argparse
import yaml
import numpy as np
import torch
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchio as tio  # For ordered patch extraction
import gc
import json

# Add the path to the helpers folder
sys.path.append('/homes/marcus/Dokumenter/LDM_precomputed_latens/helpers/')
sys.path.append('/homes/marcus/Dokumenter/LDM_precomputed_latens/')

from preprocessing import create_subjects_dataset, create_transforms_AdaIN
from models.modelvqvae_perc_adv import VQVAE3D

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Process each patch, encode to latent, compute mean/variance, and write to HDF5
def process_and_save_patches(subject, vqvae_model, patch_size, h5f, device):
    # Apply only the training transform
    training_transform, _ = create_transforms_AdaIN(patch_size)
    subject = training_transform(subject)  # Apply padding to both 'pet' and 'label'
    
    # Use GridSampler with the entire subject
    sampler = tio.data.GridSampler(subject, patch_size, patch_overlap=32)  # Reduced overlap size
    loader = DataLoader(sampler, batch_size=8, num_workers=0)  # Single batch to release memory immediately
    
    # For each patch, calculate the means and variances
    for patch in loader:
        patch_data = patch['pet'][tio.DATA].to(device)
        label_data = patch['label'][tio.DATA]

        # Encode patch into latent space
        with torch.no_grad():
            latent_patch = vqvae_model.encoder(patch_data)[0]  # Latent encoding, shape (channels, depth, height, width)

        # Compute mean and variance for each channel separately
        latent_mean = latent_patch.mean(dim=(1, 2, 3)).cpu().numpy()  # Channel-wise mean
        latent_var = latent_patch.var(dim=(1, 2, 3)).cpu().numpy()    # Channel-wise variance

        # Extract and sort unique organ labels from mask
        labels = np.unique(label_data[label_data > 0].cpu().numpy())
        label_key = str(tuple(sorted(labels)))

        # Write mean and variance to the corresponding HDF5 group immediately
        if label_key not in h5f:
            group = h5f.create_group(label_key)
            group.create_dataset("means", data=[latent_mean], maxshape=(None, latent_mean.shape[0]))
            group.create_dataset("variances", data=[latent_var], maxshape=(None, latent_var.shape[0]))
        else:
            group = h5f[label_key]
            # Resize datasets to accommodate new data
            group["means"].resize((group["means"].shape[0] + 1, latent_mean.shape[0]))
            group["variances"].resize((group["variances"].shape[0] + 1, latent_var.shape[0]))
            # Append new mean and variance
            group["means"][-1] = latent_mean
            group["variances"][-1] = latent_var
            
        # Clear memory for this patch explicitly
        del patch_data, label_data, latent_patch, latent_mean, latent_var
        torch.cuda.empty_cache()  # Optional: clear GPU memory if using CUDA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/homes/marcus/Dokumenter/LDM_precomputed_latens/config_folder/AdaIN_config.yaml')
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    # Load the config file
    config = load_config(args.config_path)

    # Set device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load the JSON file with paths
    with open('/depict/users/marcus/shared/data_splits/train_list.json', 'r') as file:
        train_list = json.load(file)

    # Limit training list
    #train_list = train_list[:500] # CHANGE LATER

    # Convert train_list to a TorchIO SubjectsDataset
    subjects_dataset = create_subjects_dataset(train_list)

    # Load the pre-trained VQ-VAE
    vqvae_model = VQVAE3D(
        in_channels=1,
        latent_dim=config['latent_dim'],
        num_embeddings=512,
        learning_rate=0.0001
    )
    vqvae_checkpoint = torch.load(config['vqvae_path'])
    # Filter out 'resnet' keys from the checkpoint
    filtered_state_dict = {k: v for k, v in vqvae_checkpoint['state_dict'].items() if not k.startswith('resnet')}
    # Load the filtered state dict into the model
    vqvae_model.load_state_dict(filtered_state_dict, strict=False)
    # Set the model to evaluation mode
    vqvae_model.to(device)
    vqvae_model.eval()
    # Freeze the model parameters
    for param in vqvae_model.parameters():
        param.requires_grad = False

    # Open HDF5 file for saving data
    save_path = "/homes/marcus/Dokumenter/LDM_precomputed_latens/AdaIN_computations/data/all_patients_V3H2O.h5"
    with h5py.File(save_path, 'a') as h5f: 
        # Process patches for each training subject and save directly to the file
        patch_size3d_train = tuple(config['patch_size3d_train'])
        for subject in tqdm(subjects_dataset):  # Iterate over subjects in the dataset
            process_and_save_patches(subject, vqvae_model, patch_size3d_train, h5f, device)
            
            # Clear memory for each subject
            del subject
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Channel-wise mean and variance data saved to {save_path}")

if __name__ == "__main__":
    main()
