import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.vector_quantizer import EMAQuantizer, VectorQuantizer
import torchio as tio
from models.resnet import resnet10  # Ensure resnet.py is accessible
from functools import partial


# Encoder using 3D convolutions for PET image patches
class VQVAEncoder3D(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(VQVAEncoder3D, self).__init__()

        self.encoder = nn.Sequential(
            # First layer: Downsample by factor of 2
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, D/2, H/2, W/2)
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            # Second layer: Downsample by factor of 2
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),           # (B, 128, D/4, H/4, W/4)
            nn.GroupNorm(16, 128),
            nn.ReLU(),

            # Third layer: Downsample by factor of 2
            nn.Conv3d(128, 128, kernel_size=4, stride=2, padding=1),          # (B, 128, D/8, H/8, W/8)
            nn.GroupNorm(16, 128),
            nn.ReLU(),

            # Final layer: Reduce to latent_dim without downsampling
            nn.Conv3d(128, latent_dim, kernel_size=3, stride=1, padding=1)    # (B, latent_dim, D/8, H/8, W/8)
        )

    def forward(self, x):
        return self.encoder(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, in_channels),  # Group normalization, 8 groups
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, in_channels)
        )

    def forward(self, x):
        return F.relu(x + self.block(x))  # Residual connection
 

# Decoder
class VQVAEDecoder3D(nn.Module):
    def __init__(self, latent_dim):
        super(VQVAEDecoder3D, self).__init__()

        # Transposed convolution layers to upsample
        self.upconv1 = nn.ConvTranspose3d(latent_dim, 128, kernel_size=4, stride=2, padding=1)  # Upsample by 2 (B, 128, D/4, H/4, W/4)
        self.res1 = ResidualBlock3D(128)  # Residual block

        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample by 2 (B, 64, D/2, H/2, W/2)
        self.res2 = ResidualBlock3D(64)  # Residual block

        self.upconv3 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample by 2 (B, 32, D, H, W)
        self.res3 = ResidualBlock3D(32)  # Residual block

        # Final convolution to reduce to a single channel output (original image size)
        self.final_conv = nn.Conv3d(32, 1, kernel_size=3, padding=1)

        self.activation = nn.Tanh()

    def forward(self, x):
        # Upsampling and residual connections
        x = self.upconv1(x)
        x = F.relu(x)
        x = self.res1(x)

        x = self.upconv2(x)
        x = F.relu(x)
        x = self.res2(x)

        x = self.upconv3(x)
        x = F.relu(x)
        x = self.res3(x)

        # Final layer
        x = self.final_conv(x)
        x = self.activation(x)
        return x


class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1, feature_maps=32):  # Reduced feature maps
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channels, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(feature_maps * 2, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        output = self.model(x)
        return output.view(-1)


class VQVAE3D(pl.LightningModule):
    def __init__(self, in_channels, latent_dim, num_embeddings, learning_rate=1e-4, lambda_perc=0.01):
        super(VQVAE3D, self).__init__()
        self.encoder = VQVAEncoder3D(in_channels, latent_dim)
        self.decoder = VQVAEDecoder3D(latent_dim)
        self.discriminator = Discriminator3D(in_channels=1)
        self.vector_quantizer = VectorQuantizer(
            quantizer=EMAQuantizer(
                spatial_dims=3,
                num_embeddings=num_embeddings,
                embedding_dim=latent_dim,
                commitment_cost=0.25,
                decay=0.9,
                epsilon=1e-5
            )
        )
        self.learning_rate = learning_rate
        self.lambda_perc = lambda_perc  # Weight for perceptual loss
        self.adv_perc = 0.01
        self.automatic_optimization = False  # Disable automatic optimization

        # Initialize ResNet for perceptual loss
        print("Initializing ResNet-10 for perceptual loss...")
        self.resnet = resnet10(
            shortcut_type='B',
            no_cuda=False
        )
        
        # Load pretrained weights
        pretrained_path = 'models/pretrained/resnet_10_23dataset.pth' 
        print(f"Loading pretrained ResNet-10 weights from {pretrained_path}...")
        checkpoint = torch.load(pretrained_path, map_location=self.device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Remove 'module.' prefix if present
            new_state_dict[name] = v
        self.resnet.load_state_dict(new_state_dict, strict=False)
        print("ResNet-10 weights loaded successfully.")

        # Freeze ResNet parameters
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        batch['pet'] = batch['pet'][tio.DATA].to(device)
        batch['label'] = batch['label'][tio.DATA].to(device)
        return batch

    def forward(self, x):
        z_e = self.encoder(x)
        vq_loss, z_q, encoding_indices = self.vector_quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, encoding_indices

    def compute_perceptual_loss(self, recon_images, target_images):
        """
        Compute perceptual loss between reconstructed and target images using ResNet-10 features.
        
        Args:
            recon_images (torch.Tensor): Reconstructed images [B, 1, D, H, W]
            target_images (torch.Tensor): Target images [B, 1, D, H, W]
        
        Returns:
            torch.Tensor: Perceptual loss value
        """
        # Extract features
        features_recon = self.resnet(recon_images)
        features_target = self.resnet(target_images)

        # Compute L1 loss between features
        perceptual_loss = 0.0
        for f_recon, f_target in zip(features_recon, features_target):
            perceptual_loss += F.mse_loss(f_recon, f_target)

        return perceptual_loss

    def training_step(self, batch, batch_idx):
        # Get optimizers
        optimizer_G, optimizer_D = self.optimizers()
        pet_patches = batch['pet'].float()

        # 
        # Train Discriminator
        # 
        # Generate reconstructions (detach to avoid gradients flowing to generator)
        recon_images, _, _ = self.forward(pet_patches)
        recon_images_detached = recon_images.detach()

        # Discriminator outputs for real and fake images
        D_real = self.discriminator(pet_patches)
        D_fake = self.discriminator(recon_images_detached)

        # Labels for real and fake images
        real_targets = torch.ones_like(D_real)
        fake_targets = torch.zeros_like(D_fake)

        # Compute discriminator loss
        loss_D_real = F.binary_cross_entropy_with_logits(D_real, real_targets)
        loss_D_fake = F.binary_cross_entropy_with_logits(D_fake, fake_targets)
        D_loss = (loss_D_real + loss_D_fake) / 2

        # Backward pass and optimization for discriminator
        optimizer_D.zero_grad()
        self.manual_backward(D_loss)
        optimizer_D.step()

        # 
        # Train Generator
        # 
        # Forward pass through the VQ-VAE
        recon_images, vq_loss, encoding_indices = self.forward(pet_patches)

        # Compute adversarial loss (generator wants discriminator to classify reconstructions as real)
        D_fake_for_G = self.discriminator(recon_images)
        adv_loss = F.binary_cross_entropy_with_logits(D_fake_for_G, real_targets)

        # Reconstruction loss
        recon_loss = F.l1_loss(recon_images, pet_patches)

        # Compute perceptual loss
        perceptual_loss = self.compute_perceptual_loss(recon_images, pet_patches)

        # Total generator loss
        G_loss = recon_loss + vq_loss + self.lambda_perc * perceptual_loss + adv_loss * self.adv_perc

        # Backward pass and optimization for generator
        optimizer_G.zero_grad()
        self.manual_backward(G_loss)
        optimizer_G.step()

        # Monitor unique codes used
        encoding_indices = encoding_indices.squeeze(-1)  # Shape: [B, D, H, W]
        flattened_indices = encoding_indices.view(-1)    # Flatten to 1D: [B * D * H * W]
        unique_codes = torch.unique(flattened_indices).size(0)
        self.log('unique_codes_train', unique_codes, sync_dist=True)

        # Logging
        self.log('D_loss', D_loss, sync_dist=True)
        self.log('G_loss', G_loss, sync_dist=True)
        self.log('recon_loss', recon_loss, sync_dist=True)
        self.log('vq_loss', vq_loss, sync_dist=True)
        self.log('perceptual_loss', perceptual_loss, sync_dist=True)
        self.log('adv_loss', adv_loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        pet_patches = batch['pet'].float()
        recon_images, vq_loss, encoding_indices = self.forward(pet_patches)

        # Reconstruction loss
        recon_loss = F.l1_loss(recon_images, pet_patches)

        # Adversarial loss
        D_fake = self.discriminator(recon_images)
        real_targets = torch.ones_like(D_fake)
        adv_loss = F.binary_cross_entropy_with_logits(D_fake, real_targets)

        # Compute perceptual loss
        perceptual_loss = self.compute_perceptual_loss(recon_images, pet_patches)

        # Total generator loss
        G_loss = recon_loss + vq_loss + self.lambda_perc * perceptual_loss + adv_loss * self.adv_perc

        # Monitor unique codes used
        encoding_indices = encoding_indices.squeeze(-1)  # Shape: [B, D, H, W]
        flattened_indices = encoding_indices.view(-1)    # Flatten to 1D: [B * D * H * W]
        unique_codes = torch.unique(flattened_indices).size(0)
        self.log('unique_codes_val', unique_codes, sync_dist=True)

        # Logging
        self.log('val_loss', G_loss, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss, sync_dist=True)
        self.log('val_vq_loss', vq_loss, sync_dist=True)
        self.log('val_perceptual_loss', perceptual_loss, sync_dist=True)
        self.log('val_adv_loss', adv_loss, sync_dist=True)
        self.log('val_unique_codes', unique_codes, sync_dist=True)
        return G_loss

    def configure_optimizers(self):
        params_G = list(self.encoder.parameters()) + \
                   list(self.decoder.parameters()) + \
                   list(self.vector_quantizer.parameters())
        optimizer_G = torch.optim.Adam(params_G, lr=self.learning_rate)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        return [optimizer_G, optimizer_D]
