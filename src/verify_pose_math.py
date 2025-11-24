import os
import numpy as np
from src.data.odometry_loader import KITTIOdometryDataset
from src.utils.device import get_config


def rotation_matrix_to_euler_angles(R):
    """
    Re-implementing here to ensure we aren't just testing the function against itself.
    Standard method for checking correctness.
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def verify_pose_math():
    cfg = get_config()

    # 1. Setup Paths
    seq_id = "00"
    pose_file = os.path.join(cfg.data.pose_path, f"{seq_id}.txt")

    if not os.path.exists(pose_file):
        print(f"[!] Error: Could not find raw pose file at {pose_file}")
        return

    print(f"[-] Reading Raw File: {pose_file}")

    # 2. MANUAL CALCULATION (Ground Truth)
    # Read first two lines only
    with open(pose_file, "r") as f:
        lines = f.readlines()
        line0 = [float(v) for v in lines[0].strip().split()]
        line1 = [float(v) for v in lines[1].strip().split()]

    # Construct Matrix 0 (Start)
    P0 = np.eye(4)
    P0[:3, :] = np.array(line0).reshape(3, 4)

    # Construct Matrix 1 (Next)
    P1 = np.eye(4)
    P1[:3, :] = np.array(line1).reshape(3, 4)

    # Calculate Delta: The movement to get from P0 to P1
    # Logic: P0 * Delta = P1  ->  Delta = inv(P0) * P1
    Delta_Matrix = np.linalg.inv(P0) @ P1

    dx_gt, dy_gt, dz_gt = Delta_Matrix[:3, 3]
    euler_gt = rotation_matrix_to_euler_angles(Delta_Matrix[:3, :3])

    print("\n[1] Ground Truth (Manual Calculation P0 -> P1):")
    print(f"    Delta XYZ: [{dx_gt:.6f}, {dy_gt:.6f}, {dz_gt:.6f}]")
    print(f"    Delta Rot: [{euler_gt[0]:.6f}, {euler_gt[1]:.6f}, {euler_gt[2]:.6f}]")

    # 3. DATASET LOADER (Test Subject)
    print("\n[-] Initializing Dataset Loader...")

    # We need to make sure we use a transform that doesn't break things,
    # but for poses it shouldn't matter.
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
            transforms.ToTensor(),
        ]
    )

    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        pose_dir=cfg.data.pose_path,
        train_sequences=[seq_id],
        seq_len=1,
        transform=tf,
    )

    # Get the first item.
    # In your logic, dataset[0] should contain the delta for Image 0 -> Image 1
    _, loaded_poses = dataset[0]
    # loaded_poses shape is (Seq_Len=1, 6), squeeze it
    loaded_pose = loaded_poses[0].numpy()

    dx_load, dy_load, dz_load = loaded_pose[:3]
    rot_load = loaded_pose[3:]

    print("\n[2] Dataset Loader Output (Item 0):")
    print(f"    Delta XYZ: [{dx_load:.6f}, {dy_load:.6f}, {dz_load:.6f}]")
    print(f"    Delta Rot: [{rot_load[0]:.6f}, {rot_load[1]:.6f}, {rot_load[2]:.6f}]")

    # 4. VERIFICATION
    print("\n[3] Comparison:")

    diff_xyz = np.abs(
        np.array([dx_gt, dy_gt, dz_gt]) - np.array([dx_load, dy_load, dz_load])
    )
    diff_rot = np.abs(euler_gt - rot_load)

    if np.all(diff_xyz < 1e-5) and np.all(diff_rot < 1e-5):
        print("\n[+] SUCCESS: Dataset logic matches manual calculation perfectly.")
    else:
        print("\n[!] FAILURE: Mismatch detected!")
        print(f"    XYZ Diff: {diff_xyz}")
        print(f"    Rot Diff: {diff_rot}")


if __name__ == "__main__":
    verify_pose_math()
