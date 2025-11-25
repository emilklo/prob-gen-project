import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset


class LatentSequenceDataset(Dataset):
    def __init__(
        self,
        processed_dir,
        seq_len=5,
        select_sequences: list[str] | None = None,
        test_sequences: list[str] | None = None,
    ):
        """
        Args:
            processed_dir: Path to folder containing '00.npz', '01.npz'
            seq_len: The window size for the RNN (e.g. 10)
            test_sequences: List of strings ['09', '10'] to exclude from loading. If None, loads all.
        """
        self.seq_len = seq_len
        self.samples = []  # List of (data_index, start_frame_index)

        self.all_mus = []
        self.all_poses = []

        # Find files

        files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))
        for seq in test_sequences or []:
            files = [f for f in files if f"{seq}.npz" not in f]
        for seq in select_sequences or []:
            files = [f for f in files if f"{seq}.npz" in f]

        current_idx = 0

        print(f"[-] Loading Latent Data from {processed_dir}...")

        for f_path in files:
            if not os.path.exists(f_path):
                raise FileNotFoundError(f"[!] Latent file not found: {f_path}")

            data = np.load(f_path)
            mu = data["mu"]  # Shape (N, 32)
            pose = data["pose"]  # Shape (N, 6)

            # Store the raw arrays in memory
            self.all_mus.append(torch.from_numpy(mu).float())
            self.all_poses.append(torch.from_numpy(pose).float())

            # Create valid window indices for THIS specific sequence
            # If sequence has 100 frames and window is 10:
            # Valid starts are 0 to 90.
            num_frames = len(mu)
            valid_starts = num_frames - seq_len + 1

            for i in range(valid_starts):
                # We store: (Which Sequence Index, Start Frame)
                self.samples.append((current_idx, i))

            current_idx += 1

        print(f"    Loaded {len(files)} sequences.")
        print(f"    Total Sliding Windows: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Get location info
        seq_idx, start_frame = self.samples[idx]

        # 2. Retrieve the full sequence array
        full_mu_seq = self.all_mus[seq_idx]
        full_pose_seq = self.all_poses[seq_idx]

        # 3. Slice the window (FAST)
        end_frame = start_frame + self.seq_len

        z_window = full_mu_seq[start_frame:end_frame]  # (Seq_Len, Latent_Dim)
        pose_window = full_pose_seq[start_frame:end_frame]  # (Seq_Len, 6)

        return z_window, pose_window


if __name__ == "__main__":
    # Simple test
    dataset = LatentSequenceDataset("data/processed_latents", seq_len=32)
    print(f"Dataset length: {len(dataset)}")
    if len(dataset) > 0:
        z, p = dataset[0]
        print(f"Sample shapes: z={z.shape}, p={p.shape}")
