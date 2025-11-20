import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class KITTIOdometryDataset(Dataset):
    def __init__(
        self, data_dir, pose_dir, sequence_id="00", seq_len=10, transform=None
    ):
        """
        Args:
            data_dir: Path to image folders (e.g., 'dataset/sequences/00/image_2')
            pose_dir: Path to pose file (e.g., 'dataset/poses/00.txt')
            seq_len: How many frames per batch (RNN needs context!)
        """
        self.seq_len = seq_len
        self.transform = transform

        # 1. Load Images
        # Assumes folder structure: data_dir/000000.png
        self.image_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".png")
            ]
        )

        # 2. Load and Process Poses
        pose_file = os.path.join(pose_dir, f"{sequence_id}.txt")
        self.pose_deltas = self._precompute_deltas(pose_file)

        # Sanity Check
        assert (
            len(self.image_files) == len(self.pose_deltas) + 1
        ), "Mismatch: N poses should generate N-1 deltas"

    def _precompute_deltas(self, pose_file):
        """
        Reads 12-float lines, converts to Matrix, calculates relative motion (Delta),
        and converts to 6D vector [x, y, z, roll, pitch, yaw].
        """
        deltas = []

        # Read all lines
        with open(pose_file, "r") as f:
            lines = f.readlines()

        # Convert to 4x4 Matrices
        poses = []
        for line in lines:
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[0:3, :] = np.array(values).reshape(3, 4)
            poses.append(pose)

        # Calculate Delta between T and T-1
        # Formula: Delta = inv(Pose_prev) @ Pose_curr
        for i in range(1, len(poses)):
            p_prev = poses[i - 1]
            p_curr = poses[i]

            # The "Step" matrix
            delta_mat = np.linalg.inv(p_prev) @ p_curr

            # Extract Translation (x,y,z)
            dx, dy, dz = delta_mat[0:3, 3]

            # Extract Rotation (Roll, Pitch, Yaw) from 3x3 submatrix
            # We use a helper function for Rotation Matrix -> Euler
            roll, pitch, yaw = self._rotation_matrix_to_euler_angles(
                delta_mat[0:3, 0:3]
            )

            deltas.append(np.array([dx, dy, dz, roll, pitch, yaw], dtype=np.float32))

        # Pad the first frame with zeros so length matches images
        # (Frame 0 has 0 movement relative to itself)
        deltas.insert(0, np.zeros(6, dtype=np.float32))

        return np.array(deltas)

    def _rotation_matrix_to_euler_angles(self, R):
        # Calculates rotation matrix to euler angles
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

    def __len__(self):
        # We return sequences. If we have 100 images and seq_len 10,
        # we can grab roughly 90 sequences.
        return len(self.image_files) - self.seq_len

    def __getitem__(self, idx):
        # Return a SEQUENCE of length 'seq_len'

        img_seq = []
        pose_seq = []

        for i in range(self.seq_len):
            # Load Image
            img_path = self.image_files[idx + i]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            img_seq.append(image)

            # Load Pose Delta
            pose_seq.append(self.pose_deltas[idx + i])

        # Stack into tensors
        # img_seq: (Seq_Len, 3, H, W)
        # pose_seq: (Seq_Len, 6)
        return torch.stack(img_seq), torch.tensor(np.array(pose_seq))
