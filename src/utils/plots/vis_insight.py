# src/utils/vis_insight.py

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from src.data.latent_loader import LatentSequenceDataset


def load_raw_kitti_poses(pose_file_path):
    """Helper to load absolute XYZ positions from a KITTI pose file."""
    poses = []
    with open(pose_file_path, "r") as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            # Reshape into 3x4 matrix
            pose = np.array(values).reshape(3, 4)
            # Extract position vector (last column: x, y, z)
            poses.append(pose[:, 3])
    return np.array(poses)


def plot_pose_insight(raw_pose_dir, processed_pose_dir, output_dir, seq_id="00"):
    """
    Generates the side-by-side comparison of absolute vs. delta poses.
    """
    print(f"\n--- Generating Pose Insight Plot for Sequence {seq_id} ---")
    raw_pose_path = Path(raw_pose_dir) / f"{seq_id}.txt"

    if not raw_pose_path.exists():
        raise FileNotFoundError(f"Raw pose file not found: {raw_pose_path}")

    # 1. Load Data
    # A. Absolute Poses (Global)
    global_poses = load_raw_kitti_poses(raw_pose_path)
    if global_poses.shape[0] == 0:
        raise ValueError(f"No poses found in {raw_pose_path}")
    global_x = global_poses[:, 0]
    # In KITTI camera frame, Y is down, Z is forward/depth.
    # Often converted so Z is up. Let's assume standard visualization mapping:
    # We usually plot X vs Z for top-down view in raw KITTI.
    global_z = global_poses[:, 2]

    # B. Pose Deltas (Local)
    # We use the dataset loader to grab the preprocessed deltas from memory
    dataset = LatentSequenceDataset(
        processed_dir=processed_pose_dir,
        seq_len=5,  # Seq len doesn't matter, we want full arrays
        select_sequences=[seq_id],
    )

    if len(dataset.all_poses) == 0:
        raise ValueError(f"No processed deltas found for sequence {seq_id}")

    # Grab the full tensor of deltas for this sequence
    local_deltas_tensor = dataset.all_poses[0]
    local_deltas = local_deltas_tensor.numpy()

    # Assuming delta format is [dx, dy, dz, ...].
    # Based on your previous descriptions, dx is forward.
    # Let's assume dz is lateral relative to car for this plot convention.
    delta_x_forward = local_deltas[:, 0]
    delta_z_lateral = local_deltas[:, 2]

    # 2. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left Panel: Absolute Poses ---
    ax1 = axes[0]
    ax1.plot(global_x, global_z, "b-", linewidth=2, label="Car Trajectory")
    ax1.scatter(global_x[0], global_z[0], c="g", s=100, marker="^", label="Start")
    ax1.scatter(global_x[-1], global_z[-1], c="r", s=100, marker="X", label="End")

    ax1.set_title("Raw Data: Absolute Poses (Global)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Global X Coordinate (meters)", fontsize=12)
    ax1.set_ylabel("Global Z Coordinate (meters)", fontsize=12)
    ax1.axis("equal")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # --- Right Panel: Pose Deltas ---
    ax2 = axes[1]
    # Use alpha to show density
    ax2.scatter(delta_x_forward, delta_z_lateral, alpha=0.4, s=5, c="purple")

    ax2.set_title("Our Approach: Pose Deltas (Local)", fontsize=14, fontweight="bold")
    ax2.set_xlabel(r"Lateral Movement ($\Delta x$) (meters/frame)", fontsize=12)
    ax2.set_ylabel(r"Forward Movement ($\Delta z$) (meters/frame)", fontsize=12)
    ax2.axis("equal")  # Important to show scale difference
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Add the insight annotation
    ax2.text(
        0.5,
        0.9,
        "STATIONARY\nDISTRIBUTION",
        transform=ax2.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="darkred",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="darkred"),
    )

    plt.tight_layout()

    output_path = Path(output_dir) / "pose_delta_insight.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved insight plot to {output_path}")
