import torch
import torch.nn as nn
import math
import torchio as tio

def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings following the implementation in Ho et al. "Denoising Diffusion Probabilistic
    Models" https://arxiv.org/abs/2006.11239.

    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        embedding_dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding


def adaptive_instance_normalization(content_features, style_mean, style_variance):
    # Ensure style_mean and style_variance are PyTorch tensors
    if not isinstance(style_mean, torch.Tensor):
        style_mean = torch.tensor(style_mean, device=content_features.device)
    if not isinstance(style_variance, torch.Tensor):
        style_variance = torch.tensor(style_variance, device=content_features.device)

    # Reshape and expand to match batch and spatial dimensions
    B, C, H, W, D = content_features.size()  # Batch size, channels, spatial dims
    style_mean = style_mean.view(B, C, 1, 1, 1)  # Match batch size and channels
    style_std = torch.sqrt(style_variance.view(B, C, 1, 1, 1) + 1e-10)

    # Compute content mean and std
    content_mean = content_features.mean(dim=(2, 3, 4), keepdim=True)
    content_std = content_features.std(dim=(2, 3, 4), keepdim=True) + 1e-10

    # Normalize content features
    normalized_content = (content_features - content_mean) / content_std

    # Apply style statistics
    adain_features = normalized_content * style_std + style_mean

    return adain_features


def sinusoidal_encoding(positions, embedding_dim):
    batch_size, num_positions = positions.shape
    assert embedding_dim % num_positions == 0, "Embedding dim must be divisible by the number of positions."

    dim_per_position = embedding_dim // num_positions
    div_term = torch.exp(torch.arange(0, dim_per_position, 2, dtype=torch.float32) * 
                         -(torch.log(torch.tensor(10000.0)) / dim_per_position))

    expanded_positions = positions.unsqueeze(-1) * div_term
    sin_encoding = torch.sin(expanded_positions)
    cos_encoding = torch.cos(expanded_positions)

    encoding = torch.cat([sin_encoding, cos_encoding], dim=-1)
    return encoding.view(batch_size, -1)


