import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# Imports
from config.logging import get_logger
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.world_model import ConvVAE
from src.utils.device import Config, get_config

logger = get_logger(__name__)


def get_available_sequences(root_dir):
    """Get list of available sequence IDs in the data directory."""
    import glob
    seq_dirs = glob.glob(os.path.join(root_dir, "*", "image_02", "data"))
    sequences = []
    for seq_dir in seq_dirs:
        seq_id = seq_dir.split(os.sep)[-3]
        if seq_id.isdigit() or (len(seq_id) == 2 and seq_id[0] == '0'):
            sequences.append(seq_id)
    return sorted(sequences)


def training(cfg: Config):
    # --- 1. Setup Configuration & Directories ---
    device = torch.device(cfg.device)

    # Create output directories
    run_name = cfg.run_name
    output_dir = f"outputs/{run_name}/rnn_checkpoints"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[-] Training RNN on {device} | Run: {run_name}")
    print(
        f"    Batch: {cfg.training.batch_size} | SeqLen: {cfg.data.rnn_sequence_length}"
    )
    print(
        f"    Layers: {cfg.model.rnn.num_layers} | Hidden: {cfg.model.rnn.hidden_size}"
    )

    # --- 2. Data & Transforms ---
    # No normalization to match VAE training (VAE decoder uses Sigmoid, expects [0,1] inputs)
    train_transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
            transforms.ToTensor(),
        ]
    )

    # --- Train/Val Split by Sequence ---
    available_sequences = get_available_sequences(cfg.data.path)
    if len(available_sequences) < 2:
        print(f"WARNING: Only {len(available_sequences)} sequences found. Using all for training.")
        train_sequences = available_sequences
        val_sequences = available_sequences
    else:
        # Use last sequence for validation
        train_sequences = available_sequences[:-1]
        val_sequences = [available_sequences[-1]]

    print(f"Train sequences: {train_sequences}")
    print(f"Val sequences: {val_sequences}")

    # Create datasets
    train_dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        train_sequences=train_sequences,
        seq_len=cfg.data.rnn_sequence_length,
        transform=train_transform,
    )

    val_dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        train_sequences=val_sequences,
        seq_len=cfg.data.rnn_sequence_length,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Train dataset size: {len(train_dataset)} samples")
    print(f"Val dataset size: {len(val_dataset)} samples")

    # --- 3. Load Pre-trained VAE (Frozen) ---
    vae = ConvVAE(
        latent_dim=cfg.model.vae.latent_dim,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
    ).to(device)

    # Use the same run_name for VAE checkpoint
    vae_path = f"outputs/{run_name}/checkpoints/vae_best.pth"
    if not os.path.exists(vae_path):
        vae_path = f"outputs/{run_name}/checkpoints/vae_final.pth"

    if os.path.exists(vae_path):
        print(f"[-] Loading VAE weights from {vae_path}")
        checkpoint = torch.load(vae_path, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("[!] CRITICAL WARNING: No VAE found. Training will fail/diverge.")

    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # --- 4. Initialize RNN ---
    rnn = DreamerMDRNN(
        latent_dim=cfg.model.vae.latent_dim,
        hidden_size=cfg.model.rnn.hidden_size,
        num_layers=cfg.model.rnn.num_layers,
        dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=float(cfg.training.learning_rate))

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 5. Training Loop ---
    best_val_loss = float("inf")
    start_epoch = 0
    best_model_path = os.path.join(output_dir, "rnn_best.pth")

    # Early stopping
    patience = 10
    patience_counter = 0
    best_epoch = 0

    LOAD_BEST = True
    if LOAD_BEST:
        if os.path.exists(best_model_path):
            print(
                f"[-] LOAD_BEST=True: Loading existing checkpoint from {best_model_path}"
            )
            checkpoint = torch.load(
                best_model_path, map_location=device, weights_only=False
            )

            # Load weights
            rnn.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load training state
            best_val_loss = checkpoint.get("val_loss", checkpoint["loss"])
            start_epoch = checkpoint["epoch"]

            print(
                f"    [*] Loaded successfully. Resuming from Epoch {start_epoch+1} with Best Val Loss: {best_val_loss:.4f}"
            )
        else:
            print(
                f"[!] LOAD_BEST=True, but {best_model_path} does not exist. Starting from scratch."
            )

    for epoch in range(start_epoch, cfg.training.epochs):
        # --- Training ---
        rnn.train()
        epoch_loss = 0
        epoch_pose_loss = 0
        num_batches = 0

        for batch_idx, (images, pose_deltas) in enumerate(train_loader):
            images = images.to(device)  # (B, Seq, 3, H, W)
            pose_targets = pose_deltas.to(device)  # (B, Seq, 6)

            # A. Get Latents
            with torch.no_grad():
                b, s, c, h, w = images.size()
                z_flat = vae.encode(images.view(b * s, c, h, w))
                z_sequence = z_flat.view(b, s, -1)

            # B. Prepare Inputs/Targets
            rnn_input = z_sequence[:, :-1, :]
            z_target = z_sequence[:, 1:, :]
            pose_target_slice = pose_targets[:, 1:, :]

            # C. Forward
            optimizer.zero_grad()
            pi, mu, sigma, pred_pose, _ = rnn(rnn_input)

            loss, l_mdn, l_pose = rnn.loss_function(
                z_target,
                pose_target_slice,
                pi,
                mu,
                sigma,
                pred_pose,
                pose_weight=1000.0,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pose_loss += l_pose.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] | "
                    f"Loss: {loss.item():.4f} | Pose: {l_pose.item():.4f}"
                )

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_pose_loss = epoch_pose_loss / num_batches if num_batches > 0 else 0

        # --- Validation ---
        rnn.eval()
        val_loss = 0
        val_pose_loss = 0
        val_batches = 0

        with torch.no_grad():
            for images, pose_deltas in val_loader:
                images = images.to(device)
                pose_targets = pose_deltas.to(device)

                # Get Latents
                b, s, c, h, w = images.size()
                z_flat = vae.encode(images.view(b * s, c, h, w))
                z_sequence = z_flat.view(b, s, -1)

                # Prepare Inputs/Targets
                rnn_input = z_sequence[:, :-1, :]
                z_target = z_sequence[:, 1:, :]
                pose_target_slice = pose_targets[:, 1:, :]

                # Forward
                pi, mu, sigma, pred_pose, _ = rnn(rnn_input)

                loss, _, l_pose = rnn.loss_function(
                    z_target,
                    pose_target_slice,
                    pi,
                    mu,
                    sigma,
                    pred_pose,
                    pose_weight=1000.0,
                )

                val_loss += loss.item()
                val_pose_loss += l_pose.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        avg_val_pose_loss = val_pose_loss / val_batches if val_batches > 0 else 0

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        print(
            f"==> Epoch {epoch+1} Complete. "
            f"Train Loss: {avg_train_loss:.4f} (Pose: {avg_pose_loss:.6f}) | "
            f"Val Loss: {avg_val_loss:.4f} (Pose: {avg_val_pose_loss:.6f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # --- Early Stopping & Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": best_val_loss,
                    "config": cfg,
                },
                best_model_path,
            )
            print(f"    [*] New Best Model Saved! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (best epoch: {best_epoch})")
                break

        # Save Periodic Model
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.training.epochs:
            save_path = os.path.join(output_dir, f"rnn_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                save_path,
            )

    print("\nRNN Training Complete.")
    print(f"Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    cfg = get_config()
    training(cfg)
