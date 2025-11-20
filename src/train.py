import argparse
import yaml
from src.utils.device import get_compute_device

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from src.models.world_model import VAE
from src.data.loaders import KITTIDataset
from src.utils.visualization import (
    save_reconstruction_comparison,
    save_latent_samples,
    plot_training_curves,
    compute_psnr
)

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Reconstruction + beta * KL Divergence
    """
    # MSE Loss (sum over all pixels)
    B = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / B
    
    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    
    return recon_loss + beta * kld, recon_loss, kld

def train_vae(cfg):
    """Training loop for VAE with reconstruction quality monitoring."""
    device = get_compute_device()
    print(f"Training VAE on {device}")

    # Config
    batch_size = cfg['training']['batch_size']
    lr = float(cfg['training']['learning_rate'])
    epochs = cfg['training']['epochs']
    latent_dim = cfg['model']['vae']['latent_dim']

    # Support both square (img_size) and rectangular (img_height, img_width)
    if 'img_size' in cfg['data']:
        img_height = cfg['data']['img_size']
        img_width = cfg['data']['img_size']
    else:
        img_height = cfg['data']['img_height']
        img_width = cfg['data']['img_width']

    # Visualization config (with defaults)
    vis_cfg = cfg.get('visualization', {})
    save_every = vis_cfg.get('save_every', 5)  # Save reconstructions every N epochs
    num_samples = vis_cfg.get('num_samples', 8)  # Number of samples to visualize

    # Create run name based on config
    if img_height == img_width:
        run_name = cfg.get('run_name', f"vae_z{latent_dim}_img{img_height}_ep{epochs}_lr{lr}_bs{batch_size}")
    else:
        run_name = cfg.get('run_name', f"vae_z{latent_dim}_img{img_height}x{img_width}_ep{epochs}")
    print(f"Run name: {run_name}")

    # Output directories based on run name
    checkpoint_dir = f"outputs/{run_name}/checkpoints"
    recon_dir = f"outputs/{run_name}/reconstructions"
    samples_dir = f"outputs/{run_name}/samples"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Save config to run directory
    import json
    with open(f"outputs/{run_name}/config.json", 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data
    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])
    dataset = KITTIDataset(root_dir=cfg['data']['path'], sequence_length=1, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Dataset size: {len(dataset)} images")
    print(f"Image size: {img_height}x{img_width}")
    if len(dataset) == 0:
        print("ERROR: No images found! Check data path and structure.")
        return

    # Model
    model = VAE(latent_dim=latent_dim, img_height=img_height, img_width=img_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss tracking
    loss_history = {'total': [], 'recon': [], 'kld': [], 'psnr': []}

    # Get a fixed batch for consistent visualization
    fixed_batch = next(iter(dataloader)).squeeze(1).to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        total_psnr = 0
        num_batches = 0

        # KL Annealing: Linear increase from 0 to 1 over first 50% of epochs
        beta = min(1.0, epoch / (epochs * 0.5))

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Batch shape: (B, 1, 3, H, W) -> squeeze to (B, 3, H, W)
            x = batch.squeeze(1).to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)

            loss, recon, kld = loss_function(recon_x, x, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            # Compute PSNR for this batch
            with torch.no_grad():
                batch_psnr = compute_psnr(x, recon_x)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            total_psnr += batch_psnr
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.2f}',
                'recon': f'{recon.item():.2f}',
                'psnr': f'{batch_psnr:.1f}',
                'beta': f'{beta:.2f}'
            })

        # Compute epoch averages
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches
        avg_psnr = total_psnr / num_batches

        # Track losses
        loss_history['total'].append(avg_loss)
        loss_history['recon'].append(avg_recon)
        loss_history['kld'].append(avg_kld)
        loss_history['psnr'].append(avg_psnr)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KLD={avg_kld:.4f}) PSNR={avg_psnr:.2f}dB")

        # Save visualizations
        if (epoch + 1) % save_every == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                fixed_recon, _, _ = model(fixed_batch)
                save_reconstruction_comparison(
                    fixed_batch, fixed_recon, epoch + 1,
                    recon_dir, num_samples=num_samples
                )
                save_latent_samples(model, 16, samples_dir, epoch + 1, device)
            model.train()
            print(f"  -> Saved reconstruction and sample visualizations")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
                'config': cfg,
            }, f"{checkpoint_dir}/vae_epoch_{epoch+1}.pth")
            print(f"  -> Saved checkpoint")

    # Save final model and training curves
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'config': cfg,
    }, f"{checkpoint_dir}/vae_final.pth")

    plot_training_curves(loss_history, f"outputs/{run_name}/training_curves.png")

    print("VAE Training Complete.")
    print(f"Final PSNR: {loss_history['psnr'][-1]:.2f}dB")
    print(f"Run directory: outputs/{run_name}/")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Visualizations saved to: {recon_dir}/")

def train_rnn(cfg):
    """Training loop for RNN (requires trained VAE)."""
    device = get_compute_device()
    print(f"Training RNN on {device}")
    # Implementation goes here
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Model components")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["vae", "rnn"], required=True, help="Component to train")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    if args.mode == "vae":
        train_vae(cfg)
    elif args.mode == "rnn":
        train_rnn(cfg)
