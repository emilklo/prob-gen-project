import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from config.logging import get_logger
from src.data.latent_loader import LatentSequenceDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.conv_vea import ConvVAE
from src.utils.device import Config, get_config
from src.utils.common import setup_run_directory
from src.utils.trajectory import evaluate_and_plot_test_sequences

logger = get_logger(__name__)


def train_mdnrnn_fast(cfg: Config):
    # --- 1. Setup & Config ---
    device = torch.device(cfg.device)

    # Define paths
    base_output_dir = Path("outputs")
    # We assume you ran the preprocess script and saved here:
    processed_data_dir = "data/processed_latents"

    # Run Naming
    if cfg.run_name == "default_run":
        run_name = (
            f"FAST_RNN_"  # Mark as the fast version
            f"{cfg.rnn.num_layers}l_"
            f"h{cfg.rnn.hidden_size}_"
            f"bs{cfg.rnn.training.batch_size}"
        )
    else:
        run_name = cfg.run_name

    run_dir = setup_run_directory(base_output_dir, run_name, cfg)
    checkpoint_dir = run_dir / "rnn_checkpoints"
    traj_dir = run_dir / "trajectories"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"[-] Training FAST MDRNN on {device}")
    print(f"    Loading pre-encoded latents from: {processed_data_dir}")

    # --- 2. The FAST Dataset ---
    # This loads .npz files instantly
    seq_len = None

    if getattr(cfg.rnn, "sequence_length", None):
        seq_len = cfg.rnn.sequence_length
    else:
        raise ValueError("Sequence length must be specified in cfg.rnn.sequence_length")

    dataset = LatentSequenceDataset(
        processed_dir=processed_data_dir,
        seq_len=seq_len,
        test_sequences=cfg.data.test_sequences,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.rnn.training.batch_size,
        shuffle=True,
        drop_last=True,  # Important for LSTM shapes
        num_workers=2,  # Low CPU overhead needed now
        pin_memory=True,
    )

    # --- 3. Initialize Models ---

    # A. The RNN (Trainable)
    # Check if num_gaussians exists in config

    rnn = DreamerMDRNN(
        latent_dim=cfg.vae.latent_dim,
        hidden_size=cfg.rnn.hidden_size,
        num_layers=cfg.rnn.num_layers,
    ).to(device)

    optimizer = optim.Adam(rnn.parameters(), lr=float(cfg.rnn.training.learning_rate))

    # B. The VAE (Visuals Only)
    # We only need this to decode "Dreams" into images for the plot.
    # It is NOT used in the training loop.

    # --- 4. Resume Logic (Optional) ---
    best_loss = float("inf")
    start_epoch = 0
    best_model_path = base_output_dir / "rnn_best.pth"
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

    # --- 5. Training Loop ---
    print(f"[-] Starting Training Loop for {cfg.rnn.training.epochs} epochs...")

    for epoch in range(start_epoch, cfg.rnn.training.epochs):
        rnn.train()
        epoch_loss = 0
        epoch_mdn_loss = 0
        epoch_pose_loss = 0

        # NEW: Loop gives us Latents directly! No images.
        # z_seq: (Batch, Seq_Len, Latent_Dim)
        # pose_seq: (Batch, Seq_Len, 6)
        for batch_idx, (z_seq, pose_seq) in enumerate(dataloader):

            z_seq = z_seq.to(device)
            pose_seq = pose_seq.to(device)

            # --- SLICING LOGIC ---
            # Input:  Steps 0 to N-1
            # Target: Steps 1 to N

            # 1. RNN Input: The latent vector at time t
            rnn_input = z_seq[:, :-1, :]

            # 2. Vision Target: The latent vector at time t+1
            z_target = z_seq[:, 1:, :]

            # 3. Pose Target: The movement that occurs between t and t+1
            # Recall dataset logic: pose[t] IS the delta from t -> t+1
            # So we want the pose aligned with the Input.
            pose_target = pose_seq[:, :-1, :]

            # Forward
            optimizer.zero_grad()

            # rnn returns: pi, mu, sigma, pred_pose, hidden
            pi, mu, sigma, pred_pose, _ = rnn(rnn_input)

            # Loss
            loss, l_mdn, l_pose = rnn.loss_function(
                y_true_latent=z_target,
                y_true_pose=pose_target,
                pi=pi,
                mu=mu,
                sigma=sigma,
                pred_pose=pred_pose,
                pose_weight=100.0,  # Adjust based on your magnitude analysis
            )

            loss.backward()

            # Gradient Clipping (Crucial for LSTMs)
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_mdn_loss += l_mdn.item()
            epoch_pose_loss += l_pose.item()

            if batch_idx % 100 == 0:
                print(
                    f"Ep {epoch+1} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} (MDN: {l_mdn.item():.2f}, Pose: {l_pose.item():.4f})"
                )

        # --- End Epoch Stats ---
        avg_loss = epoch_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        # --- Saving & Evaluation ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                checkpoint_dir / "rnn_best.pth",
            )
            print("    [*] Best Model Saved")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Save regular checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": rnn.state_dict(),
                },
                checkpoint_dir / f"rnn_epoch_{epoch+1}.pth",
            )

            # Run Visualization (Uses VAE to decode dreams)
            if hasattr(cfg.data, "test_sequences") and cfg.data.test_sequences:
                print("    [*] Dreaming Trajectories on Test Set...")
                evaluate_and_plot_test_sequences(
                    model=rnn,
                    vae=None,
                    test_sequences=cfg.data.test_sequences,
                    cfg=cfg,
                    save_dir=traj_dir,
                    epoch=epoch + 1,
                    device=device,
                )


if __name__ == "__main__":
    cfg = get_config()
    train_mdnrnn_fast(cfg)
