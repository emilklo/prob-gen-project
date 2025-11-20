import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Imports
from config.logging import get_logger
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.world_model import ConvVAE
from src.utils.device import Config, get_config

logger = get_logger(__name__)


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
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
            transforms.ToTensor(),
        ]
    )

    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        seq_len=cfg.data.rnn_sequence_length,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # --- 3. Load Pre-trained VAE (Frozen) ---
    vae = ConvVAE(
        latent_dim=cfg.model.vae.latent_dim,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
    ).to(device)

    # TODO: Add 'vae_checkpoint' to your config file to avoid hardcoding
    vae_path = "outputs/vae_z64_img128_mps/checkpoints/vae_final.pth"

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
        num_layers=cfg.model.rnn.num_layers,  # <--- From Config
    ).to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=float(cfg.training.learning_rate))

    # --- 5. Training Loop ---
    best_loss = float("inf")
    start_epoch = 0
    best_model_path = os.path.join(output_dir, "rnn_best.pth")
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
            best_loss = checkpoint["loss"]
            start_epoch = checkpoint["epoch"]

            print(
                f"    [*] Loaded successfully. Resuming from Epoch {start_epoch+1} with Best Loss: {best_loss:.4f}"
            )
        else:
            print(
                f"[!] LOAD_BEST=True, but {best_model_path} does not exist. Starting from scratch."
            )

    for epoch in range(start_epoch, cfg.training.epochs):
        rnn.train()
        epoch_loss = 0
        epoch_pose_loss = 0

        for batch_idx, (images, pose_deltas) in enumerate(dataloader):
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

            if batch_idx % 50 == 0:
                print(
                    f"Epoch {epoch+1} [{batch_idx}/{len(dataloader)}] | "
                    f"Loss: {loss.item():.4f} | Pose: {l_pose.item():.4f}"
                )

        # --- End of Epoch Stats ---
        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")

        # --- SAVE CHECKPOINTS ---
        # 1. Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(output_dir, "rnn_best.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "config": cfg,
                },
                save_path,
            )
            print(f"    [*] New Best Model Saved!")

        # 2. Save Periodic/Final Model
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.training.epochs:
            save_path = os.path.join(output_dir, f"rnn_epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_path,
            )


if __name__ == "__main__":
    cfg = get_config()
    training(cfg)
