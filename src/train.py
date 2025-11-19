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
    """Training loop for VAE."""
    device = get_compute_device()
    print(f"Training VAE on {device}")
    
    # Config
    batch_size = cfg['training']['batch_size']
    lr = float(cfg['training']['learning_rate'])
    epochs = cfg['training']['epochs']
    latent_dim = cfg['model']['vae']['latent_dim']
    img_size = cfg['data']['img_size']
    
    # Data
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    dataset = KITTIDataset(root_dir=cfg['data']['path'], sequence_length=1, transform=transform) # Seq len 1 for VAE
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = VAE(latent_dim=latent_dim, img_size=img_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        # KL Annealing: Linear increase from 0 to 1 over first 50% of epochs
        beta = min(1.0, epoch / (epochs * 0.5))
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Batch shape: (B, 1, 3, 64, 64) -> squeeze to (B, 3, 64, 64)
            x = batch.squeeze(1).to(device)
            
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            
            loss, recon, kld = loss_function(recon_x, x, mu, logvar, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            
            pbar.set_postfix({'loss': loss.item(), 'beta': beta})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Recon={total_recon/len(dataloader):.4f}, KLD={total_kld/len(dataloader):.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/vae_epoch_{epoch+1}.pth")
            
    print("VAE Training Complete.")
    torch.save(model.state_dict(), "checkpoints/vae_final.pth")

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
