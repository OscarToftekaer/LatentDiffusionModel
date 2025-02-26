#!/usr/bin/env python3

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import yaml
import argparse
import wandb
import os
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import json
#########################################################
#from models.modelvqvae import VQVAE3D
#from training_module.lightning import VQVAE3DLightning
#########################################################
from models.modelvqvaenospade import VQVAE3D
#from models.modelvqvaenospadeextralayer import VQVAE3D

from helpers.preprocessing import create_subjects_dataset, create_queue, create_transforms



def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml')
    parser.add_argument('--wandb_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=4)
    parser.add_argument('--wandb_project', type=str, required=True)

    args = parser.parse_args()

    # Load the config file
    config = load_config(args.config_path)

    # Update config with command-line hyperparameters if provided
    if args.wandb_name is not None:
        config['wandb_name'] = args.wandb_name
    if args.wandb_project is not None:
        config['wandb_project'] = args.wandb_project

    # Set the seed for reproducibility
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize WandB logger only in the main process
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        wandb_logger = WandbLogger(project=config['wandb_project'], name=config['wandb_name'])
    else:
        wandb_logger = None

    # Load the lists from JSON files
    with open('data_splits/train_list.json', 'r') as f:
        train_list = json.load(f)

    with open('data_splits/val_list.json', 'r') as f:
        val_list = json.load(f)
    
    # Dictionary of class probabilities
    probabilities_dict = {i: 1 if i != 0 else 0 for i in range(config['num_classes'])}
    probabilities_dict[90] = 10 # increase probability for sampling patch for brain
    
    # Create transforms
    patch_size3d_train = tuple(config['patch_size3d_train'])
    patch_size3d_val = tuple(config['patch_size3d_val'])

    training_transforms, validation_transforms = create_transforms(patch_size3d_train, patch_size3d_val)

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



    # Define the checkpoint callback
    best_model_checkpoint = ModelCheckpoint(
        monitor='val_loss',        # Monitor validation loss
        dirpath='checkpoints/',    # Directory to save the checkpoints
        filename='V5Adversarial_PerceptualBrain-{epoch:02d}-{val_loss:.8f}',  # Filename format
        save_top_k=2,              # Save the top model with the best validation loss
        mode='min',                # We want to minimize the validation loss
        save_weights_only=True     # Save only the model weights
    )


    # Define the checkpoint callback for saving every 5 epochs
    epoch_checkpoint = ModelCheckpoint(
    every_n_epochs=2,          # Save every 2 epochs
    dirpath='checkpoints/',    # Directory to save the checkpoints
    filename='V5EpochModel-{epoch:02d}-{val_loss:.8f}',
    monitor='val_loss',
    save_top_k = -1,  # Filename format
    save_weights_only=True     # Save only the model weights
    )




    ###########################################################################################
    # No spade normalization
    vqvae_model = VQVAE3D(in_channels=config['in_channels'], 
                          latent_dim=config['latent_dim'], 
                          num_embeddings=config['num_embeddings'], 
                          learning_rate=config['learning_rate'],
                          lambda_perc=0.01)
    ###########################################################################################


    # Trainer
    # trainer = pl.Trainer(
    #     max_epochs=config['max_epochs'],
    #     logger=wandb_logger,
    #     accelerator="gpu",
    #     devices=torch.cuda.device_count(),
    #     strategy="ddp",  # DDP strategy for distributed training
    #     #sync_batchnorm=True,  # Synchronize batch norm layers across GPUs
    #     log_every_n_steps=1,
    #     callbacks=[checkpoint_callback]
    # )

    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        logger=wandb_logger,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy=DDPStrategy(find_unused_parameters=True),
        log_every_n_steps=1,
        callbacks=[best_model_checkpoint,epoch_checkpoint]
    )

    # Training
    trainer.fit(vqvae_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Finalize TorchIO queues
    queue_train = None
    queue_val = None

    # Clean up distributed processes
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()



    wandb.finish()  # Ensure that the current W&B run is properly closed

if __name__ == "__main__":
    main()
