from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Imports
from config.logging import get_logger
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.conv_vea import ConvVAE
from src.utils.device import Config, get_config

logger = get_logger(__name__)


def train_mdnrnn(cfg: Config):
    # --- 1. Setup Configuration & Directories ---
    device = torch.device(cfg.device)

    # --- 2. Data & Transforms ---
    # Initialize dataset EARLY to use its properties for run naming
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
            transforms.ToTensor(),
        ]
    )

    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        seq_len=cfg.rnn.sequence_length,
        transform=transform,
        pose_dir=cfg.data.pose_path,
        test_sequences=cfg.data.test_sequences,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.rnn.training.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # --- 3. Run Naming & Directory Setup ---
    base_output_dir = Path("outputs")

    # Create run name based on config
    if cfg.run_name == "default_run":
        run_name = (
            f"mdnrnn_rnn{cfg.rnn.num_layers}l_"
            f"h{cfg.rnn.hidden_size}_"
            f"bs{cfg.rnn.training.batch_size}_"
            f"seq{cfg.rnn.sequence_length}_"
            f"numseq{dataset.num_sequences()}"
        )
    else:
        run_name = cfg.run_name

    from src.utils.common import setup_run_directory

    run_dir = setup_run_directory(base_output_dir, run_name, cfg)

    output_dir = run_dir / "rnn_checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directory for trajectory plots
    traj_dir = run_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"[-] Training RNN on {device} | Run: {cfg.run_name}")
    print(
        f"    Batch: {cfg.rnn.training.batch_size} | SeqLen: {cfg.rnn.sequence_length}"
    )
    print(f"    Layers: {cfg.rnn.num_layers} | Hidden: {cfg.rnn.hidden_size}")
    if cfg.data.test_sequences:
        print(f"    Test Sequences: {cfg.data.test_sequences}")

    # --- 3. Load Pre-trained VAE (Frozen) ---
    vae = ConvVAE(
        latent_dim=cfg.vae.latent_dim,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
    ).to(device)

    # TODO: Add 'vae_checkpoint' to your config file to avoid hardcoding
    vae_path = Path("outputs/vae_z64_img128_mps2/checkpoints/vae_epoch_20.pth")

    if vae_path.exists():
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
        latent_dim=cfg.vae.latent_dim,
        hidden_size=cfg.rnn.hidden_size,
        num_layers=cfg.rnn.num_layers,  # <--- From Config
    ).to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=float(cfg.rnn.training.learning_rate))

    # --- 5. Training Loop ---
    best_loss = float("inf")
    start_epoch = 0
    best_model_path = output_dir / "rnn_best.pth"
    LOAD_BEST = True
    if LOAD_BEST:
        if best_model_path.exists():
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

    from src.utils.trajectory import evaluate_and_plot_test_sequences

    for epoch in range(start_epoch, cfg.rnn.training.epochs):
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
            # (Batch, Seq_Len-1, Latent_Dim)
            rnn_input = z_sequence[:, :-1, :]
            z_target = z_sequence[:, 1:, :]

            # (Batch, Seq_Len-1, 6)
            # New Alignment: pose_deltas[t] is movement t -> t+1
            # So for input z[t], we want action delta[t]
            pose_target_slice = pose_targets[:, :-1, :]

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
            save_path = output_dir / "rnn_best.pth"
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
            print("    [*] New Best Model Saved!")

        # 2. Save Periodic/Final Model
        if (epoch + 1) % 5 == 0 or (epoch + 1) == cfg.rnn.training.epochs:
            save_path = output_dir / f"rnn_epoch_{epoch+1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                save_path,
            )

            # 3. Evaluate on Test Sequences
            if cfg.data.test_sequences:
                evaluate_and_plot_test_sequences(
                    model=rnn,
                    vae=vae,
                    test_sequences=cfg.data.test_sequences,
                    cfg=cfg,
                    save_dir=traj_dir,
                    epoch=epoch + 1,
                    device=device,
                )


if __name__ == "__main__":
    cfg = get_config()
    train_mdnrnn(cfg)
