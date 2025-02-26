#!/usr/bin/env python3

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from models.ldm_pinaya import DiffusionModelLightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies import DDPStrategy
import argparse
import yaml
import wandb


class LatentLabelDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Collect all label files
        self.label_files = sorted([f for f in os.listdir(folder_path) if f.startswith("labels_") and f.endswith(".npy")])
        # Collect all latent files
        self.latent_files = sorted([f for f in os.listdir(folder_path) if f.startswith("latent_") and f.endswith(".npy")])
        # Ensure corresponding label and latent files exist
        assert len(self.label_files) == len(self.latent_files), "Mismatch in label and latent files"
        for label_file, latent_file in zip(self.label_files, self.latent_files):
            assert label_file.split('_')[1] == latent_file.split('_')[1], \
                f"File mismatch: {label_file} and {latent_file}"

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        label_path = os.path.join(self.folder_path, self.label_files[idx])
        latent_path = os.path.join(self.folder_path, self.latent_files[idx])
        
        # Load the data
        labels = np.load(label_path, allow_pickle=True)  # Labels are small arrays
        latent = np.load(latent_path).astype(np.float32)  # Convert to float32 for PyTorch compatibility

        # Convert `labels` to a list
        labels = labels.astype(int).tolist() # Ensure labels are a list, not a NumPy array
        
        return {"labels": labels, "latent": latent}
    
def custom_collate_fn(batch):
    # Collect latents and labels
    latents = torch.stack([torch.tensor(item["latent"], dtype=torch.float32) for item in batch])  # Stack latents
    labels = [item["labels"] for item in batch]  # Keep labels as lists
    return {"labels": labels, "latents": latents}

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config_folder/config_ldm_multiGPU.yaml')
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, required=True)

    args = parser.parse_args()

    # Load the config file
    config = load_config(args.config_path)

    # Update config with command-line hyperparameters if provided
    if args.wandb_name is not None:
        config['wandb_name'] = args.wandb_name
    if args.wandb_project is not None:
        config['wandb_project'] = args.wandb_project

    # Initialize WandB logger only in the main process
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        wandb_logger = WandbLogger(project=config['wandb_project'], name=config['wandb_name'])
    else:
        wandb_logger = None


    # Dataset and DataLoader
    train_set = LatentLabelDataset(config["train_folder"])
    val_set = LatentLabelDataset(config["val_folder"])

    ## TESTING ##
    #train_subset_size = 16
    #val_subset_size = 4
    #train_set = Subset(train_set, list(range(min(train_subset_size, len(train_set)))))
    #val_set = Subset(val_set, list(range(min(val_subset_size, len(val_set)))))
    #############

    train_loader = DataLoader(train_set, batch_size=config["batch_size_train"], shuffle=True, num_workers=config["num_workers"], collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=config["batch_size_val"], shuffle=False, num_workers=config["num_workers"], collate_fn=custom_collate_fn)

    # Model Initialization
    model = DiffusionModelLightning(
        in_channels=config["latent_dim"],
        out_channels=config["latent_dim"],
        h5_file_path=config["h5_path"],
        lr=config["learning_rate"]
    )

    # Define the checkpoint callback
    checkpoint_best = ModelCheckpoint(
        monitor='val_loss',        # Monitor validation loss
        dirpath='/depict/users/marcus/shared/LDM_precomputed_latens_checkpoints/ldm_h2o_pinaya_it3/',    # Directory to save the checkpoints
        filename='h2o_LDM_pinaya_it3-best-{epoch:02d}-{val_loss:.8f}',
        save_top_k=1,              # Save the top model with the best validation loss
        mode='min',                # We want to minimize the validation loss
        save_weights_only=True     # Save only the model weights
    )

    # Save checkpoints every 3 epochs
    checkpoint_periodic = ModelCheckpoint(
        dirpath='/depict/users/marcus/shared/LDM_precomputed_latens_checkpoints/ldm_h2o_pinaya_it3/',
        filename='h2o_LDM_pinaya_it3-periodic-{epoch:02d}',
        every_n_epochs=1,
        save_top_k=-1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    

    # Determine if DDP should be enabled based on GPU count
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = 'auto'  # Default single GPU/no DDP


    trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator="gpu",
            logger=wandb_logger,
            devices=torch.cuda.device_count(),
            log_every_n_steps=1,
            callbacks=[checkpoint_best, checkpoint_periodic, lr_monitor],
            strategy=strategy #find_unused_parameters=True
        )
    trainer.fit(model, train_loader, val_loader)


    # Clean up distributed processes
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    wandb.finish()  # Ensure that the current W&B run is properly closed




if __name__ == "__main__":
    main()