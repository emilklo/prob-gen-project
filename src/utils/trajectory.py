import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.world_model import ConvVAE
import torchvision.transforms as transforms
from src.utils.device import get_compute_device, get_config


def euler_to_matrix(roll, pitch, yaw):
    """
    Converts Euler angles to a 3x3 Rotation Matrix.
    KITTI uses specific order, but standard Rz*Ry*Rx usually works well for vis.
    """
    # Rx (Roll)
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    # Ry (Pitch)
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    # Rz (Yaw)
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    return Rz @ Ry @ Rx


def integrate_path(deltas_6d):
    """
    Takes a list of 6D deltas [dx, dy, dz, r, p, y]
    and accumulates them into global [x, y, z] coordinates.
    """
    current_pose = np.eye(4)  # Start at Identity (0,0,0)
    path = [[0, 0, 0]]

    for delta in deltas_6d:
        dx, dy, dz, r, p, y = delta

        # 1. Create Local Transformation Matrix (Step)
        # Translation
        T_local = np.eye(4)
        T_local[0:3, 3] = [dx, dy, dz]
        # Rotation
        R_local = euler_to_matrix(r, p, y)
        T_local[0:3, 0:3] = R_local

        # 2. Update Global Pose
        # Global_New = Global_Old * Local_Step
        current_pose = current_pose @ T_local

        # 3. Extract new global position
        path.append(current_pose[0:3, 3])

    return np.array(path)


def run_evaluation():
    device = get_compute_device()
    cfg = get_config()

    img_height, img_width = cfg.data.img_height, cfg.data.img_width

    # 1. Load Data (Validation Sequence, e.g., 00 or 07)
    # No normalization to match training (VAE expects [0,1] inputs)
    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )

    # Note: We use seq_len=5 to match training configuration
    # Sequence 04 is completely unseen by MDN-RNN (train: 00, val: 10)
    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        train_sequences=["04"],  # Evaluate on Sequence 04 (completely unseen, no gaps)
        seq_len=5,  # Match training sequence length
        transform=transform,
    )

    # 2. Load Models
    print("[-] Loading models...")
    latent_dim = cfg.model.vae.latent_dim
    hidden_size = cfg.model.rnn.hidden_size
    num_layers = cfg.model.rnn.num_layers

    vae = ConvVAE(latent_dim=latent_dim, img_height=img_height, img_width=img_width).to(device)
    rnn = DreamerMDRNN(latent_dim=latent_dim, hidden_size=hidden_size, num_layers=num_layers).to(device)

    # --- PATHS (UPDATE THESE) ---
    run_name = cfg.run_name
    vae_path = f"outputs/{run_name}/checkpoints/vae_final.pth"
    rnn_path = f"outputs/{run_name}/rnn_checkpoints/rnn_best.pth"

    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=False)["model_state_dict"])
    rnn.load_state_dict(torch.load(rnn_path, map_location=device, weights_only=False)["model_state_dict"])

    vae.eval()
    rnn.eval()

    # 3. Run Inference Loop
    print("[-] predicting trajectory...")

    pred_deltas = []
    true_deltas = []

    # We assume the dataset loader provided gives us access to the raw lists
    # If not, we iterate the standard way.
    # Let's iterate 500 frames (50 seconds of driving)

    limit = min(500, len(dataset))
    hidden = None  # LSTM hidden state

    with torch.no_grad():
        # Process sequences to match training configuration
        for i in range(limit):
            imgs, poses = dataset[i]  # Returns seq_len=5

            # imgs: (5, 3, H, W)
            # poses: (5, 6)

            # Encode all frames in the sequence
            imgs_batch = imgs.unsqueeze(0).to(device)  # (1, 5, 3, H, W)
            b, s, c, h, w = imgs_batch.size()
            z_flat = vae.encode(imgs_batch.view(b * s, c, h, w))
            z_sequence = z_flat.view(b, s, -1)  # (1, 5, latent_dim)

            # RNN expects input sequence (we use all but last to predict transitions)
            rnn_input = z_sequence[:, :-1, :]  # (1, 4, latent_dim)

            # RNN Step - feed full sequence context
            _, _, _, pred_pose, hidden = rnn(rnn_input, hidden)

            # pred_pose shape: (1, 4, 6) - predictions for transitions 0->1, 1->2, 2->3, 3->4
            # Use the last prediction which has most context
            pred_d = pred_pose[0, -1].cpu().numpy()
            true_d = poses[-1].cpu().numpy()  # Target is the last delta

            pred_deltas.append(pred_d)
            true_deltas.append(true_d)

            if i % 50 == 0:
                print(f"Processed {i}/{limit} frames")

    # 4. Integrate Paths (The Math Part)
    print("[-] Integrating paths...")
    path_pred = integrate_path(pred_deltas)
    path_true = integrate_path(true_deltas)

    # 5. Plot
    plt.figure(figsize=(10, 10))
    # KITTI Coordinates: X is Right, Z is Forward.
    # We plot X vs Z to get the Bird's Eye View.
    plt.plot(path_true[:, 0], path_true[:, 2], "k-", label="Ground Truth", linewidth=2)
    plt.plot(path_pred[:, 0], path_pred[:, 2], "r--", label="Ours (RNN)", linewidth=2)

    plt.title(f"Visual Odometry Result (Seq 04, {limit} frames)")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.axis("equal")  # Crucial to see turns correctly
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file = "trajectory_result_new.png"
    plt.savefig(out_file)
    print(f"[-] Saved plot to {out_file}")


if __name__ == "__main__":
    run_evaluation()
