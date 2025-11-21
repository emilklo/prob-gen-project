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

    def __init__(self, root_dir: str, sequence_length: int = 1, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset (e.g., data/kitti).
            sequence_length (int): Number of frames per sequence.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform

        # Find all sequence directories
        sequence_dirs = sorted(
            [d for d in glob.glob(os.path.join(root_dir, "*")) if os.path.isdir(d)]
        )

        self.sequences = []
        for seq_dir in sequence_dirs:
            # Check for 'data' subdirectory
            image_dir = os.path.join(seq_dir, "image_02", "data")
            if not os.path.exists(image_dir):
                image_dir = os.path.join(seq_dir, "image_02")

            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            if len(image_files) == 0:
                continue
            if len(image_files) >= sequence_length:
                self.sequences.append(image_files)

        if not self.sequences:
            logger.warning(
                "No valid sequences found in %s. Check dataset structure.", root_dir
            )

        # Calculate the total number of possible sequences
        self.num_sequences = sum(
            len(seq) - sequence_length + 1 for seq in self.sequences
        )

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Find which sequence and which starting frame this index corresponds to
        seq_paths = []
        for seq_images in self.sequences:
            num_possible_starts = len(seq_images) - self.sequence_length + 1
            if idx < num_possible_starts:
                # This is the correct sequence
                start_frame = idx
                seq_paths = seq_images[start_frame : start_frame + self.sequence_length]
                break
            idx -= num_possible_starts

        images = []
        for p in seq_paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # For VAE training, we want a single image, not a sequence of length 1
        if self.sequence_length == 1:
            return images[0]

        # Stack into (seq_len, C, H, W)
        return torch.stack(images)
