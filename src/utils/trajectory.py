from pathlib import Path
import torch
import numpy as np

from src.utils.device import (
    Config,
    get_config,
    get_compute_device,
)
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Import your modules
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.world_model import ConvVAE
from src.utils.common import get_unique_path


def euler_to_matrix(roll, pitch, yaw):
    """Converts Euler angles to a 3x3 Rotation Matrix."""
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx


def integrate_path(deltas_6d):
    """Accumulates 6D deltas into global [x, y, z] coordinates."""
    current_pose = np.eye(4)
    path = [np.zeros(3)]

    for delta in deltas_6d:
        dx, dy, dz, r, p, y = delta

        # Create Local Transformation Matrix (Step)
        T_local = np.eye(4)
        T_local[0:3, 3] = [dx, dy, dz]
        T_local[0:3, 0:3] = euler_to_matrix(r, p, y)

        # Update Global Pose
        current_pose = current_pose @ T_local

        # Extract new global position
        path.append(current_pose[0:3, 3])

    return np.array(path)


def evaluate_and_plot_test_sequences(
    model: DreamerMDRNN,
    vae: ConvVAE,
    test_sequences: list[str],
    cfg: Config,
    save_dir: Path,
    epoch: int,
    device: torch.device,
):
    """
    Evaluates the model on each test sequence and plots the trajectory.
    """
    print(f"[-] Evaluating on test sequences: {test_sequences}")

    # Setup transform
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
            transforms.ToTensor(),
        ]
    )

    model.eval()
    vae.eval()

    for seq_id in test_sequences:
        print(f"    Processing Sequence {seq_id}...")

        # Create a temporary dataset for this sequence
        try:
            dataset = KITTIOdometryDataset(
                root_dir=cfg.data.path,
                pose_dir=cfg.data.pose_path,
                train_sequences=[seq_id],  # Only load this specific sequence
                seq_len=cfg.rnn.sequence_length,
                transform=transform,
            )
        except Exception as e:
            print(f"    [!] Failed to load sequence {seq_id}: {e}")
            continue

        if len(dataset) == 0:
            print(f"    [!] Sequence {seq_id} is empty or invalid.")
            continue

        pred_deltas = []
        true_deltas = []
        hidden = None

        # Limit frames for faster plotting during training if needed,
        # but usually we want the full test sequence.
        limit = len(dataset)

        with torch.no_grad():
            for i in range(limit):
                imgs, poses = dataset[i]  # imgs shape: (Seq, 3, H, W)

                # Prepare Input (Frame t)
                # We only need the first frame of the sequence window to predict the next step
                # But wait, the RNN needs a sequence.
                # Actually, for trajectory generation, we usually feed:
                # z_t -> RNN -> z_t+1, pose_delta_t
                # Here we are doing open-loop or closed-loop?
                # The previous code did: z = vae.encode(img_t0) -> rnn(z, hidden)
                # This implies we are feeding GROUND TRUTH images at each step (Open Loop for images),
                # but accumulating the predicted pose deltas.

                img_t0 = imgs[0].unsqueeze(0).to(device)

                # Encode
                z = vae.encode(img_t0).unsqueeze(1)  # Shape: (1, 1, Latent)

                # RNN Step
                pi, mu, sigma, pred_pose, hidden = model(z, hidden)

                # Extract Prediction (Movement t -> t+1)
                pred_d = pred_pose[0, 0].cpu().numpy()

                # Extract Ground Truth (Delta stored at index 1 of the window)
                # The dataset returns a window of poses.
                # poses[0] is pose at t, poses[1] is pose at t+1.
                # The delta we want is the movement FROM t TO t+1.
                # The dataset's 'sequence_deltas' are precomputed deltas.
                # dataset[i] returns (images, pose_seq).
                # pose_seq contains [delta_t, delta_t+1, ...].
                # So pose_seq[0] is the delta for the first step in this window.

                # Wait, let's check KITTIOdometryDataset.__getitem__
                # It returns `pose_seq = all_deltas[start_frame : start_frame + self.seq_len]`
                # So poses[0] IS the delta for the current frame t.

                true_d = poses[0].cpu().numpy()

                pred_deltas.append(pred_d)
                true_deltas.append(true_d)

        # Integrate
        path_pred = integrate_path(pred_deltas)
        path_true = integrate_path(true_deltas)

        # Plot
        plt.figure(figsize=(10, 10))
        plt.plot(
            path_true[:, 0], path_true[:, 2], "k-", label="Ground Truth", linewidth=2
        )
        plt.plot(
            path_pred[:, 0], path_pred[:, 2], "r--", label="Ours (RNN)", linewidth=2
        )

        plt.title(f"Trajectory Result (Seq {seq_id}, Epoch {epoch})")
        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        plt.axis("equal")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = get_unique_path(
            save_dir / f"trajectory_seq{seq_id}_epoch{epoch}.png"
        )
        plt.savefig(save_path)
        plt.close()
        print(f"    Saved plot to {save_path}")


def main():
    # 1. Define Model Path (Hardcoded for standalone execution)
    # You might want to make these arguments or read from a specific run config
    rnn_path = Path("outputs/mdnrnn_1/rnn_checkpoints/rnn_best.pth")
    vae_path = Path("outputs/vae_z128_img128x416_ep100/checkpoints/vae_epoch_40.pth")

    # 2. Load Configuration from Model Path
    run_dir = rnn_path.parent.parent
    config_path = run_dir / "config.json"

    print(f"[-] Loading Config from {config_path}")
    if config_path.exists():

        cfg = Config.from_file(config_path)

        # Override device with current best available if needed, or trust config?
        # Usually we want to use the best available device for inference
        device = get_compute_device()
        print(f"    Config loaded. Run Name: {cfg.run_name}")
    else:
        print(f"[!] Config not found at {config_path}. Falling back to default.")
        cfg = get_config()
        device = get_compute_device()

    print("[-] Evaluation Config Loaded.")
    print(f"    Latent Dim: {cfg.vae.latent_dim}")
    print(f"    RNN Layers: {cfg.rnn.num_layers}")

    # 3. Initialize Models
    vae = ConvVAE(
        latent_dim=cfg.vae.latent_dim,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width,
    ).to(device)

    rnn = DreamerMDRNN(
        latent_dim=cfg.vae.latent_dim,
        hidden_size=cfg.rnn.hidden_size,
        num_layers=cfg.rnn.num_layers,
    ).to(device)

    print(f"[-] Loading VAE: {vae_path}")
    if vae_path.exists():
        vae.load_state_dict(
            torch.load(vae_path, map_location=device)["model_state_dict"]
        )
    else:
        print(f"[!] Warning: VAE checkpoint not found at {vae_path}")

    print(f"[-] Loading RNN: {rnn_path}")
    if rnn_path.exists():
        rnn.load_state_dict(
            torch.load(rnn_path, map_location=device, weights_only=False)[
                "model_state_dict"
            ]
        )
    else:
        print(f"[!] Warning: RNN checkpoint not found at {rnn_path}")

    # 4. Run Evaluation
    save_dir = run_dir / "trajectories"
    save_dir.mkdir(parents=True, exist_ok=True)

    test_sequences = cfg.data.test_sequences

    evaluate_and_plot_test_sequences(
        model=rnn,
        vae=vae,
        test_sequences=test_sequences,
        cfg=cfg,
        save_dir=save_dir,
        epoch=999,  # Indicating standalone run
        device=device,
    )


if __name__ == "__main__":
    main()
