import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class KITTIOdometryDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        pose_dir: str = "data/kitti/poses",
        train_sequences: list | None = None,
        test_sequences: list | None = None,
        seq_len: int = 10,
        transform=None,
    ):
        """
        Args:
            root_dir: Path to sequences folder (e.g., 'dataset/sequences')
            pose_dir: Path to poses folder (e.g., 'dataset/poses')
            train_sequences: List of strings ['00', '01', '02'...]
            test_sequences: List of strings to EXCLUDE from training (e.g. ['09', '10'])
            seq_len: Length of RNN sequence window.
        """
        self.root_dir = root_dir
        self.pose_dir = pose_dir
        self.seq_len = seq_len
        self.transform = transform
        # Default to KITTI training set if none provided
        if train_sequences is None:
            self.train_sequences = [
                "00",
                "01",
                "02",
                "03",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                # "10", test set!
            ]
        else:
            self.train_sequences = train_sequences

        # Filter out test sequences
        if test_sequences:
            original_count = len(self.train_sequences)
            self.train_sequences = [
                s for s in self.train_sequences if s not in test_sequences
            ]
            removed_count = original_count - len(self.train_sequences)
            if removed_count > 0:
                print(
                    f"    [Dataset] Excluded {removed_count} test sequences: {test_sequences}"
                )

        # Master lists to store data references
        self.sequence_images = {}  # Map: '00' -> [path1, path2...]
        self.sequence_deltas = {}  # Map: '00' -> numpy array of deltas
        self.valid_samples = []  # List of tuples: ('00', start_index)

        # print(f"[-] Loading KITTI Odometry Dataset...")

        for seq_id in self.train_sequences:
            self._load_single_sequence(seq_id)

        print(f"    Total Valid Sequences: {len(self.valid_samples)}")

    def _load_single_sequence(self, seq_id):
        """
        Loads one specific drive (e.g., 00), calculates its poses,
        and adds valid sliding-window indices to the master list.
        """
        # 1. Find Images
        img_dir_with_data = os.path.join(self.root_dir, seq_id, "image_02", "data")
        img_paths = sorted(glob.glob(os.path.join(img_dir_with_data, "*.png")))

        if not img_paths:
            # Fallback to the non-'data' directory structure
            img_dir_without_data = os.path.join(self.root_dir, seq_id, "image_02")
            img_paths = sorted(glob.glob(os.path.join(img_dir_without_data, "*.png")))

        if not img_paths:
            print(f"Warning: No images found for sequence {seq_id}")
            return

        # 2. Load Poses & Calculate Deltas
        pose_file = os.path.join(self.pose_dir, f"{seq_id}.txt")
        if not os.path.exists(pose_file):
            print(f"Warning: No pose file found for {seq_id} at {pose_file}")
            return

        deltas = self._precompute_deltas(pose_file)

        # Sanity Check: We have N-1 deltas for N images.
        # The delta[i] corresponds to the transform from frame i to i+1.
        # So, we should associate delta[i] with image[i+1].
        # We trim the first image to align them.
        if len(img_paths) > len(deltas):
            img_paths = img_paths[1 : len(deltas) + 1]

        min_len = min(len(img_paths), len(deltas))
        img_paths = img_paths[:min_len]
        deltas = deltas[:min_len]

        # 3. Store in Memory
        self.sequence_images[seq_id] = img_paths
        self.sequence_deltas[seq_id] = deltas

        # 4. Create Valid Sliding Windows
        # If seq has 100 frames and seq_len is 10, we can start at 0...90
        num_valid_starts = min_len - self.seq_len + 1

        for i in range(num_valid_starts):
            self.valid_samples.append((seq_id, i))

    def _precompute_deltas(self, pose_file):
        """
        Reads 12-float absolute poses, converts to 6-float relative deltas.
        """
        with open(pose_file, "r") as f:
            lines = f.readlines()

        poses = []
        for line in lines:
            values = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[0:3, :] = np.array(values).reshape(3, 4)
            poses.append(pose)

        deltas = []
        # Calculate Delta: inv(Pose_t-1) @ Pose_t
        for i in range(1, len(poses)):
            p_prev = poses[i - 1]
            p_curr = poses[i]
            delta_mat = np.linalg.inv(p_prev) @ p_curr

            dx, dy, dz = delta_mat[0:3, 3]
            roll, pitch, yaw = self._rotation_matrix_to_euler_angles(
                delta_mat[0:3, 0:3]
            )
            deltas.append(np.array([dx, dy, dz, roll, pitch, yaw], dtype=np.float32))

        # # Pad first frame (zero movement)
        # deltas.insert(0, np.zeros(6, dtype=np.float32)) # we trimmed the first image, so no need to pad
        return np.array(deltas)

    def _rotation_matrix_to_euler_angles(self, R):
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
        return len(self.valid_samples)

    def num_sequences(self):
        return len(self.sequence_images)

    def __getitem__(self, idx):
        # 1. Get Metadata for this specific window
        seq_id, start_frame = self.valid_samples[idx]

        # 2. Retrieve actual data pointers
        all_imgs = self.sequence_images[seq_id]
        all_deltas = self.sequence_deltas[seq_id]

        # 3. Slice the Window
        img_paths = all_imgs[start_frame : start_frame + self.seq_len]
        pose_seq = all_deltas[start_frame : start_frame + self.seq_len]

        # 4. Load Images
        images = []
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Return (Seq_Len, 3, H, W) and (Seq_Len, 6)
        return torch.stack(images), torch.tensor(np.array(pose_seq))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.utils.device import get_config

    # 1. Load Configuration
    cfg = get_config()
    print("[-] Configuration loaded.")

    # 2. Define Transform (Crucial: PIL -> Tensor)
    # Resizing to standard KITTI size or smaller for testing

    img_height = cfg.data.img_height
    img_width = cfg.data.img_width

    tf = transforms.Compose(
        [transforms.Resize((img_height, img_width)), transforms.ToTensor()]
    )

    # 3. Initialize Dataset
    try:
        dataset = KITTIOdometryDataset(
            root_dir=cfg.data.path,
            train_sequences=None,  # Default to all
            seq_len=cfg.rnn.sequence_length,
            transform=tf,
            pose_dir=cfg.data.pose_path,
        )
    except KeyError as e:
        print(f"Error: Missing key in YAML file: {e}")
        exit(1)

    # 4. Test Data Loading
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n[-] Testing Batch Generation...")
    try:
        # Fetch one batch
        images, poses = next(iter(loader))

        print(f"    Batch Size:      {batch_size}")
        print(f"    Sequence Length: {cfg.rnn.sequence_length}")
        print("-" * 30)
        print(f"    Image Tensor Shape: {images.shape}")
        # Expected: (Batch, Seq_Len, Channels, Height, Width)

        print(f"    Pose Tensor Shape:  {poses.shape}")
        # Expected: (Batch, Seq_Len, 6)

        print("-" * 30)
        print("    Data Range Check:")
        print(f"    Images (min/max): {images.min():.2f} / {images.max():.2f}")
        print(f"    Poses  (mean):    {poses.mean():.4f}")

        print("\n[+] Test Successful.")

    except Exception as e:
        print(f"\n[!] Test Failed: {e}")
        # Print full traceback for debugging
        import traceback

        traceback.print_exc()
