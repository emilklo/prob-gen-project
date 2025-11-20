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
    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )

    # Note: We set seq_len=1 because we will manually loop through the frames
    # to simulate a continuous drive.
    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        train_sequences=["00"],  # Evaluate on Sequence 00
        seq_len=2,  # Minimal context needed to grab pairs
        transform=transform,
    )

    # 2. Load Models
    print("[-] Loading models...")
    vae = ConvVAE(latent_dim=128, img_height=img_height, img_width=img_width).to(device)
    rnn = DreamerMDRNN(latent_dim=128, hidden_size=512).to(device)

    # --- PATHS (UPDATE THESE) ---
    vae_path = "outputs/vae_z64_img128_mps/checkpoints/vae_final.pth"
    rnn_path = "outputs/rnn_checkpoints/rnn_final.pth"

    vae.load_state_dict(torch.load(vae_path, map_location=device)["model_state_dict"])
    rnn.load_state_dict(torch.load(rnn_path, map_location=device)["model_state_dict"])

    vae.eval()
    rnn.eval()

    # 3. Run Inference Loop
    print("[-] predicting trajectory...")

    pred_deltas = []
    true_deltas = []

    # We assume the dataset loader provided gives us access to the raw lists
    # If not, we iterate the standard way.
    # Let's iterate 500 frames (50 seconds of driving)

    limit = 500
    hidden = None  # LSTM hidden state

    with torch.no_grad():
        # To simulate streaming, we feed frames one by one
        # We grab window [t, t+1] from dataset
        for i in range(limit):
            imgs, poses = dataset[i]  # Returns seq_len=2

            # imgs: (2, 3, H, W)
            # poses: (2, 6) -> We want movement from 0->1

            img_t0 = imgs[0].unsqueeze(0).to(device)  # (1, 3, H, W)

            # Encode
            z = vae.encode(img_t0).unsqueeze(1)  # (1, 1, 128)

            # RNN Step
            # We feed z_t. The RNN predicts the transition and next state
            pi, mu, sigma, pred_pose, hidden = rnn(z, hidden)

            # Pred_pose shape: (1, 1, 6)
            pred_d = pred_pose[0, 0].cpu().numpy()
            true_d = poses[1].cpu().numpy()  # Target is the delta at index 1

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

    plt.title(f"Visual Odometry Result (Seq 00, {limit} frames)")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.axis("equal")  # Crucial to see turns correctly
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file = "trajectory_result.png"
    plt.savefig(out_file)
    print(f"[-] Saved plot to {out_file}")


if __name__ == "__main__":
    run_evaluation()
