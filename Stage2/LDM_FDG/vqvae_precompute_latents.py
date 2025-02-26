#!/usr/bin/env python3

import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchio as tio  # For ordered patch extraction
import json



# Paths for inputs
sys.path.append('/homes/marcus/Dokumenter/LDM_fdg/helpers/')
sys.path.append('/homes/marcus/Dokumenter/LDM_fdg/')
config_path = '/homes/marcus/Dokumenter/LDM_fdg/config_folder/vqvae_precompute_config.yaml'
train_path = '/depict/users/marcus/shared/data_splits/train_list.json'
val_path = '/depict/users/marcus/shared/data_splits/val_list.json'

train_output_folder = "/depict/users/marcus/shared/latents_fdg_finedtuned/train"
val_output_folder = "/depict/users/marcus/shared/latents_fdg_finedtuned/val"

#train_output_folder = "/homes/marcus/Dokumenter/LDM_fdg/train"
#val_output_folder = "/homes/marcus/Dokumenter/LDM_fdg/val"


from preprocessing import create_subjects_dataset, create_transforms_AdaIN, create_queue
from models.modelvqvae_perc_adv import VQVAE3D

# Load configuration file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_start_index(output_folder):
    """Find the starting index for new files in the output folder."""
    existing_files = os.listdir(output_folder)
    max_index = -1
    for filename in existing_files:
        if filename.startswith("latent_") and filename.endswith(".npy"):
            try:
                # Extract the number from the filename
                index = int(filename.split('_')[1].split('.')[0])
                max_index = max(max_index, index)
            except ValueError:
                continue
    return max_index + 1  # Start from the next available index


def compute_latents(dataloader, vqvae_model, output_folder, device):
    """
    Process patches through the VQ-VAE encoder, extract label combinations, and save latents.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get the starting index
    start_index = get_start_index(output_folder)
    
    for i, batch in enumerate(tqdm(dataloader, desc=f"Processing {output_folder}")):
        # Extract patch data and move to device
        patch_data = batch['pet'][tio.DATA].to(device).float()  # Assuming 'pet' is the key for input data
        label_data = batch['label'][tio.DATA]  # Extract label data from the batch (not moved to GPU)

        with torch.no_grad():
            latent_patches = vqvae_model.encoder(patch_data)

        for idx, latent in enumerate(latent_patches):
            # Calculate the file index
            file_index = start_index + i * len(latent_patches) + idx

            # Extract the unique labels (non-zero) for the current patch
            unique_labels = torch.unique(label_data[idx][label_data[idx] > 0]).cpu().numpy()
            label_combination = tuple(sorted(unique_labels))  # Ensure consistency in ordering

            # Save the latent encoding to a file
            latent_file = os.path.join(output_folder, f"latent_{file_index}.npy")
            np.save(latent_file, latent.cpu().numpy())  # Save latent to CPU before saving

            # Save the label combination alongside
            label_file = os.path.join(output_folder, f"labels_{file_index}.npy")
            np.save(label_file, label_combination)  # Save label combination

            # Save the label combination alongside
            seg_data = label_data[idx]
            seg_file = os.path.join(output_folder, f"seg_{file_index}.npy")
            np.save(seg_file, seg_data)  # Save segmentation mask
            

        # Clear memory
        del patch_data, label_data, latent_patches
        torch.cuda.empty_cache()



def main():
    # Load the config file
    config = load_config(config_path)
    # Set device for computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    with open(train_path, 'r') as file:
        train_list = json.load(file)
    with open(val_path, 'r') as file:
        val_list = json.load(file)
    # TEMP
    #train_list = train_list[:2]
    #val_list = val_list[:2]

    # Dictionary of class probabilities
    probabilities_dict = {i: 1 if i != 0 else 0 for i in range(config['num_classes'])}

    # Create transforms
    patch_size3d_train = tuple(config['patch_size3d_train'])
    patch_size3d_val = tuple(config['patch_size3d_val'])

    training_transforms, validation_transforms = create_transforms_AdaIN(patch_size3d_train, patch_size3d_val)

    # Data preparation
    dataset_train = create_subjects_dataset(train_list, transforms=training_transforms)
    queue_train = create_queue(
        dataset_train, patch_size3d_train, probabilities_dict,
        max_length=config['max_length_train'],
        samples_per_volume=config['samples_per_volume_train']
    )
    train_dataloader = DataLoader(queue_train, batch_size=config['batch_size_train'], num_workers=0, persistent_workers=False)

    dataset_val = create_subjects_dataset(val_list, transforms=validation_transforms)
    queue_val = create_queue(
        dataset_val, patch_size3d_val, probabilities_dict,
        max_length=config['max_length_val'],
        samples_per_volume=config['samples_per_volume_val'],
        shuffle_subjects=False, shuffle_patches=False
    )
    val_dataloader = DataLoader(queue_val, batch_size=config['batch_size_val'], num_workers=0, persistent_workers=False)


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

    # Save latents
    compute_latents(train_dataloader, vqvae_model, train_output_folder, device)
    compute_latents(val_dataloader, vqvae_model, val_output_folder, device)

    print("Latents saved successfully.")



if __name__ == "__main__":
    main()