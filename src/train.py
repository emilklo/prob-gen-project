import argparse
from dataclasses import asdict
from pathlib import Path
from src.utils.device import get_config, Config
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from src.utils.common import setup_run_directory
from src.train_mdnrnn import train_mdnrnn


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


def train_vae(cfg: Config):
    """Training loop for VAE with reconstruction quality monitoring."""
    # cfg.device is a string, but we can also use get_compute_device() for the torch object
    # or just torch.device(cfg.device)
    device = torch.device(cfg.device)
    print(f"Training VAE on {device}")

    # Config
    batch_size = cfg.vae.training.batch_size
    lr = cfg.vae.training.learning_rate
    epochs = cfg.vae.training.epochs
    latent_dim = cfg.vae.latent_dim

    # Support both square (img_size) and rectangular (img_height, img_width)

    img_height = cfg.data.img_height
    img_width = cfg.data.img_width

    # Visualization config
    save_every = cfg.visualization.save_every
    num_samples = cfg.visualization.num_samples

    # Data
    transform = transforms.Compose(
        [transforms.Resize((img_height, img_width)), transforms.ToTensor()]
    )
    dataset = KITTIDataset(
        root_dir=cfg.data.path, sequence_length=1, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # number of sequences in the dataset
    num_seq = len(dataset.sequences)

    # Create run name based on config
    if cfg.run_name == "default_run":  # Check if it's the default fallback or from yaml
        if img_height == img_width:
            run_name = f"vae_z{latent_dim}_img{img_height}_ep{epochs}_lr{lr}_bs{batch_size}_seq{num_seq}"
        else:
            run_name = (
                f"vae_z{latent_dim}_img{img_height}x{img_width}_ep{epochs}_seq{num_seq}"
            )
    else:
        run_name = cfg.run_name

    print(f"Run name: {run_name}")

    # Output directories based on run name
    base_output_dir = Path("outputs")

    run_dir = setup_run_directory(base_output_dir, run_name, cfg)

    checkpoint_dir = run_dir / "checkpoints"
    recon_dir = run_dir / "reconstructions"
    samples_dir = run_dir / "samples"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    # Config is already saved by setup_run_directory

    print(f"Dataset size: {len(dataset)} images")
    print(f"Image size: {img_height}x{img_width}")
    if len(dataset) == 0:
        print("ERROR: No images found! Check data path and structure.")
        return

    # Model
    model = ConvVAE(
        latent_dim=latent_dim, img_height=img_height, img_width=img_width
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss tracking
    loss_history = {"total": [], "recon": [], "kld": [], "psnr": []}

    # Get a fixed batch for consistent visualization
    try:
        fixed_batch = next(iter(dataloader)).squeeze(1).to(device)
    except StopIteration:
        print("ERROR: Dataloader is empty.")
        return

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

        # Track losses
        loss_history["total"].append(avg_loss)
        loss_history["recon"].append(avg_recon)
        loss_history["kld"].append(avg_kld)
        loss_history["psnr"].append(avg_psnr)

        print(
            f"Epoch {epoch+1}: Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, KLD={avg_kld:.4f}) PSNR={avg_psnr:.2f}dB"
        )

        # Save visualizations
        if (epoch + 1) % save_every == 0 or epoch == 0:
            model.eval()
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
            model.train()
            print("  -> Saved reconstruction and sample visualizations")

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
            print("  -> Saved checkpoint")

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

    print("VAE Training Complete.")
    print(f"Final PSNR: {loss_history['psnr'][-1]:.2f}dB")
    print(f"Run directory: outputs/{run_name}/")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Visualizations saved to: {recon_dir}/")


def train_rnn(cfg: Config):
    """Training loop for RNN (requires trained VAE)."""
    train_mdnrnn(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train World Model components")
    # Removed --config argument as we now use automatic config loading
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
