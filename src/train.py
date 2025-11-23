import argparse
import json
from dataclasses import asdict
from src.utils.device import get_config, Config

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms
from tqdm import tqdm

from src.models.world_model import ConvVAE
from src.data.loaders import KITTIDataset
from src.utils.visualization import (
    save_reconstruction_comparison,
    save_latent_samples,
    plot_training_curves,
    compute_psnr,
)


def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Reconstruction + beta * KL Divergence
    """
    # MSE Loss (sum over all pixels)
    B = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / B

    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B

    return recon_loss + beta * kld, recon_loss, kld


def get_available_sequences(root_dir):
    """Get list of available sequence IDs in the data directory."""
    import glob
    seq_dirs = glob.glob(os.path.join(root_dir, "*", "image_02", "data"))
    sequences = []
    for seq_dir in seq_dirs:
        # Extract sequence ID from path
        seq_id = seq_dir.split(os.sep)[-3]
        if seq_id.isdigit() or (len(seq_id) == 2 and seq_id[0] == '0'):
            sequences.append(seq_id)
    return sorted(sequences)


def train_vae(cfg: Config):
    """Training loop for VAE with reconstruction quality monitoring."""
    device = torch.device(cfg.device)
    print(f"Training VAE on {device}")

    # Config
    batch_size = cfg.training.batch_size
    lr = cfg.training.learning_rate
    epochs = cfg.training.epochs
    latent_dim = cfg.model.vae.latent_dim

    img_height = cfg.data.img_height
    img_width = cfg.data.img_width

    # Visualization config
    save_every = cfg.visualization.save_every
    num_samples = cfg.visualization.num_samples

    # Create run name based on config
    if cfg.run_name == "default_run":
        if img_height == img_width:
            run_name = (
                f"vae_z{latent_dim}_img{img_height}_ep{epochs}_lr{lr}_bs{batch_size}"
            )
        else:
            run_name = f"vae_z{latent_dim}_img{img_height}x{img_width}_ep{epochs}"
    else:
        run_name = cfg.run_name

    print(f"Run name: {run_name}")

    # Output directories based on run name
    checkpoint_dir = f"outputs/{run_name}/checkpoints"
    recon_dir = f"outputs/{run_name}/reconstructions"
    samples_dir = f"outputs/{run_name}/samples"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Save config to run directory
    with open(f"outputs/{run_name}/config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- Data Transforms with Augmentation ---
    # Note: No normalization for VAE training because decoder uses Sigmoid (outputs 0-1)
    # Normalization is only used for RNN training where VAE encoder is frozen
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )

    # --- Train/Val Split by Sequence ---
    # VAE can use all images (gaps don't matter for single-image training)
    # Train: 00, 01, 02, 03, 10 (all available)
    # Val: 04 (held out for MDN-RNN inference)
    train_sequences = ["00", "01", "02", "03", "10"]
    val_sequences = ["04"]

    print(f"Train sequences: {train_sequences}")
    print(f"Val sequences: {val_sequences}")

    # Create datasets
    train_dataset = KITTIDataset(
        root_dir=cfg.data.path, sequence_length=1, transform=train_transform, sequences=train_sequences
    )
    val_dataset = KITTIDataset(
        root_dir=cfg.data.path, sequence_length=1, transform=val_transform, sequences=val_sequences
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train dataset size: {len(train_dataset)} images")
    print(f"Val dataset size: {len(val_dataset)} images")
    print(f"Image size: {img_height}x{img_width}")

    if len(train_dataset) == 0:
        print("ERROR: No training images found! Check data path and structure.")
        return

    # Model
    model = ConvVAE(latent_dim=latent_dim, img_height=img_height, img_width=img_width).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss tracking
    loss_history = {"total": [], "recon": [], "kld": [], "psnr": [], "val_loss": [], "val_psnr": []}

    # Get a fixed batch for consistent visualization
    try:
        fixed_batch = next(iter(val_loader)).squeeze(1).to(device)
    except StopIteration:
        print("ERROR: Validation dataloader is empty.")
        return

    # Early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        total_recon = 0
        total_kld = 0
        total_psnr = 0
        num_batches = 0

        # KL Annealing: Linear increase from 0 to 1 over first 50% of epochs
        beta = min(1.0, epoch / (epochs * 0.5))

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            x = batch.squeeze(1).to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)

            loss, recon, kld = loss_function(recon_x, x, mu, logvar, beta=beta)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_psnr = compute_psnr(x, recon_x)

            total_loss += loss.item()
            total_recon += recon.item()
            total_kld += kld.item()
            total_psnr += batch_psnr
            num_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.2f}",
                    "recon": f"{recon.item():.2f}",
                    "psnr": f"{batch_psnr:.1f}",
                    "beta": f"{beta:.2f}",
                }
            )

        # Compute epoch averages
        avg_loss = total_loss / num_batches
        avg_recon = total_recon / num_batches
        avg_kld = total_kld / num_batches
        avg_psnr = total_psnr / num_batches

        # --- Validation ---
        model.eval()
        val_loss = 0
        val_psnr = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                x = batch.squeeze(1).to(device)
                recon_x, mu, logvar = model(x)
                loss, _, _ = loss_function(recon_x, x, mu, logvar, beta=1.0)

                val_loss += loss.item()
                val_psnr += compute_psnr(x, recon_x)
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_psnr = val_psnr / val_batches if val_batches > 0 else 0

        # Track losses
        loss_history["total"].append(avg_loss)
        loss_history["recon"].append(avg_recon)
        loss_history["kld"].append(avg_kld)
        loss_history["psnr"].append(avg_psnr)
        loss_history["val_loss"].append(avg_val_loss)
        loss_history["val_psnr"].append(avg_val_psnr)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} PSNR={avg_psnr:.2f}dB | "
            f"Val Loss={avg_val_loss:.4f} PSNR={avg_val_psnr:.2f}dB | LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_history": loss_history,
                    "config": asdict(cfg),
                },
                f"{checkpoint_dir}/vae_best.pth",
            )
            print(f"  -> New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (best epoch: {best_epoch})")
                break

        # Save visualizations
        if (epoch + 1) % save_every == 0 or epoch == 0:
            with torch.no_grad():
                fixed_recon, _, _ = model(fixed_batch)
                save_reconstruction_comparison(
                    fixed_batch,
                    fixed_recon,
                    epoch + 1,
                    recon_dir,
                    num_samples=num_samples,
                )
                save_latent_samples(model, 16, samples_dir, epoch + 1, device)
            print(f"  -> Saved reconstruction and sample visualizations")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_history": loss_history,
                    "config": asdict(cfg),
                },
                f"{checkpoint_dir}/vae_epoch_{epoch+1}.pth",
            )
            print(f"  -> Saved checkpoint")

    # Save final model and training curves
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": asdict(cfg),
        },
        f"{checkpoint_dir}/vae_final.pth",
    )

    plot_training_curves(loss_history, f"outputs/{run_name}/training_curves.png")

    print("\nVAE Training Complete.")
    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Final Train PSNR: {loss_history['psnr'][-1]:.2f}dB")
    print(f"Final Val PSNR: {loss_history['val_psnr'][-1]:.2f}dB")
    print(f"Run directory: outputs/{run_name}/")
    print(f"Best checkpoint: {checkpoint_dir}/vae_best.pth")


def train_rnn(cfg: Config):
    """Training loop for RNN (requires trained VAE)."""
    device = torch.device(cfg.device)
    print(f"Training RNN on {device}")
    # Implementation goes here
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Model components")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["vae", "rnn"],
        required=True,
        help="Component to train",
    )

    args = parser.parse_args()

    # Load config automatically
    cfg = get_config()

    if args.mode == "vae":
        train_vae(cfg)
    elif args.mode == "rnn":
        train_rnn(cfg)
