import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image

from config.logging import get_logger

logger = get_logger(__name__)


class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI image sequences.
    Expected structure: root_dir/<sequence_id>/image_02/data/*.png
    """

    def __init__(
        self, root_dir: str, sequence_length: int = 1, transform=None, sequences: list = None
    ):
        """
        Args:
            root_dir (str): Path to the dataset (e.g., data/kitti).
            sequence_length (int): Number of frames per sequence.
            transform (callable, optional): Transform to apply to images.
            sequences (list, optional): List of sequence IDs to include (e.g., ["00", "01"]).
                                        If None, uses all available sequences.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Find all image files
        # Assuming structure: root/00/image_02/data/*.png
        if sequences is None:
            self.image_files = sorted(
                glob.glob(os.path.join(root_dir, "*", "image_02", "data", "*.png"))
            )
        else:
            self.image_files = []
            for seq_id in sequences:
                seq_files = sorted(
                    glob.glob(os.path.join(root_dir, seq_id, "image_02", "data", "*.png"))
                )
                self.image_files.extend(seq_files)

        if not self.image_files:
            logger.warning(
                "No images found in %s. Make sure structure is root/seq/image_02/data/",
                root_dir,
            )

        # We can create sequences starting from index 0 to len - seq_len
        self.num_sequences = max(0, len(self.image_files) - sequence_length + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get sequence of file paths
        seq_paths = self.image_files[idx : idx + self.sequence_length]

        images = []
        for p in seq_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Stack into (seq_len, C, H, W)
        return torch.stack(images)
