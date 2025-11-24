import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.conv_vea import ConvVAE
from src.models.mdnrnn_pose import DreamerMDRNN
from src.utils.device import Config, get_config, get_compute_device
from src.utils.common import get_unique_path

#


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
    vae: ConvVAE | None,
    test_sequences: list[str],
    cfg: Config,
    save_dir: Path,
    epoch: int,
    device: torch.device,
    limit: int = 1000,
    unique_plots: bool = True,
):
    """
    Evaluates the model trajectory using the deterministic Pose Head.
    """
    print(f"[-] Evaluating on: {test_sequences}")

    # Locate data
    processed_dir = Path("data/processed_latents")
    if not processed_dir.exists():
        processed_dir = Path(cfg.data.path) / "processed_latents"

    model.eval()

    for seq_id in test_sequences:
        npz_path = processed_dir / f"{seq_id}.npz"
        if not npz_path.exists():
            continue

        print(f"    Processing Sequence {seq_id}...")

        try:
            data = np.load(npz_path)
            mus = torch.from_numpy(data["mu"]).float().to(device)
            poses = data["pose"]
        except Exception as e:
            print(f"    [!] Error loading {npz_path}: {e}")
            continue

        pred_deltas = []
        true_deltas = []
        hidden = None

        num_steps = min(len(mus) - 1, limit)

        with torch.no_grad():
            for t in range(num_steps):
                # Input: Current latent z_t
                z_t = mus[t].unsqueeze(0).unsqueeze(0)  # (1, 1, 32)

                # Forward Pass
                # We ignore the MDN outputs (pi, mu, sigma) because they represent
                # the *next latent image*, not the movement.
                # We take 'pred_pose' which is the movement.
                _, _, _, pred_pose, hidden = model(z_t, hidden)

                # Extract Prediction
                # pred_pose shape is (Batch=1, Seq=1, 6)
                pred_d = pred_pose[0, 0].cpu().numpy()

                # Ground Truth
                true_d = poses[t]

                pred_deltas.append(pred_d)
                true_deltas.append(true_d)

        # Integrate and Plot
        path_pred = integrate_path(pred_deltas)
        path_true = integrate_path(true_deltas)

        plt.figure(figsize=(10, 10))
        plt.plot(
            path_true[:, 0], path_true[:, 2], "k-", label="Ground Truth", linewidth=2
        )
        plt.plot(
            path_pred[:, 0], path_pred[:, 2], "r--", label="Ours (RNN)", linewidth=2
        )

        plt.title(f"Trajectory (Seq {seq_id}, Epoch {epoch})")
        plt.xlabel("X (meters)")
        plt.ylabel("Z (meters)")
        plt.axis("equal")
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = (
            get_unique_path(save_dir / f"trajectory_seq{seq_id}_epoch{epoch}.png")
            if unique_plots
            else save_dir / f"trajectory_seq{seq_id}_epoch{epoch}.png"
        )
        plt.savefig(save_path)
        plt.close()
        print(f"    Saved plot to {save_path}")


def main():
    # 1. Configuration
    # Path to where your '00.npz', '01.npz' files are stored
    processed_latents_dir = Path("data/processed_latents")

    rnn_path = Path("outputs/FAST_RNN_1l_h256_bs512/rnn_checkpoints/rnn_best.pth")

    # We assume config lies relative to the model path
    run_dir = rnn_path.parent.parent
    config_path = run_dir / "config.json"

    # 2. Load Config
    print(f"[-] Loading Config from {config_path}")
    if config_path.exists():
        cfg = Config.from_file(config_path)
        device = get_compute_device()
    else:
        print(f"[!] Config not found at {config_path}. Falling back to default.")
        cfg = get_config()
        device = get_compute_device()

    print(f"    Latent Dim: {cfg.vae.latent_dim}")
    print(f"    RNN Layers: {cfg.rnn.num_layers}")

    # 3. Initialize RNN
    # Note: VAE is NOT needed for trajectory eval since we load latents directly!
    rnn = DreamerMDRNN(
        latent_dim=cfg.vae.latent_dim,
        hidden_size=cfg.rnn.hidden_size,
        num_layers=cfg.rnn.num_layers,
    ).to(device)

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

    # Ensure we have sequences to test (e.g. ['09', '10'])
    test_sequences = cfg.data.test_sequences
    if not test_sequences:
        test_sequences = ["09", "10"]  # Fallback

    evaluate_and_plot_test_sequences(
        model=rnn,
        vae=None,  # VAE not needed here
        test_sequences=test_sequences,
        cfg=cfg,
        save_dir=save_dir,
        epoch=999,  # Indicating standalone run
        device=device,
    )


if __name__ == "__main__":
    main()
