"""
Visualization utilities for VAE training and evaluation.
"""
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def save_reconstruction_comparison(original, reconstruction, epoch, save_dir, num_samples=8):
    """
    Save a side-by-side comparison of original and reconstructed images.

    Args:
        original: Original images tensor (B, C, H, W)
        reconstruction: Reconstructed images tensor (B, C, H, W)
        epoch: Current epoch number
        save_dir: Directory to save images
        num_samples: Number of samples to show
    """
    os.makedirs(save_dir, exist_ok=True)

    # Take first num_samples
    n = min(num_samples, original.size(0))
    orig = original[:n].cpu()
    recon = reconstruction[:n].cpu()

    # Create grid: top row = original, bottom row = reconstruction
    comparison = torch.cat([orig, recon], dim=0)
    grid = make_grid(comparison, nrow=n, normalize=True, padding=2)

    # Convert to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    # Plot and save
    fig, ax = plt.subplots(figsize=(2 * n, 4))
    ax.imshow(grid_np)
    ax.set_title(f'Epoch {epoch}: Original (top) vs Reconstruction (bottom)')
    ax.axis('off')

    save_path = os.path.join(save_dir, f'reconstruction_epoch_{epoch:04d}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    return save_path


def save_latent_samples(model, num_samples, save_dir, epoch, device):
    """
    Generate and save samples from random latent vectors.

    Args:
        model: VAE model
        num_samples: Number of samples to generate
        save_dir: Directory to save images
        epoch: Current epoch number
        device: Torch device
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Sample random latent vectors
        z = torch.randn(num_samples, model.latent_dim).to(device)

        # Decode using the model's decode method
        samples = model.decode(z)

        # Create grid
        grid = make_grid(samples.cpu(), nrow=int(np.sqrt(num_samples)), normalize=True, padding=2)
        grid_np = grid.permute(1, 2, 0).numpy()

        # Plot and save
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid_np)
        ax.set_title(f'Epoch {epoch}: Random Samples from Latent Space')
        ax.axis('off')

        save_path = os.path.join(save_dir, f'samples_epoch_{epoch:04d}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)

    model.train()
    return save_path


def plot_training_curves(losses, save_path):
    """
    Plot and save training loss curves.

    Args:
        losses: Dictionary with keys 'total', 'recon', 'kld', each containing list of epoch losses
        save_path: Path to save the plot
    """
    epochs = range(1, len(losses['total']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    axes[0].plot(epochs, losses['total'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[1].plot(epochs, losses['recon'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss (MSE)')
    axes[1].grid(True, alpha=0.3)

    # KL Divergence
    axes[2].plot(epochs, losses['kld'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return save_path


def compute_psnr(original, reconstruction):
    """
    Compute Peak Signal-to-Noise Ratio between original and reconstructed images.

    Args:
        original: Original images tensor (B, C, H, W) in [0, 1]
        reconstruction: Reconstructed images tensor (B, C, H, W) in [0, 1]

    Returns:
        Mean PSNR across batch
    """
    mse = torch.mean((original - reconstruction) ** 2, dim=[1, 2, 3])
    psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
    return psnr.mean().item()
