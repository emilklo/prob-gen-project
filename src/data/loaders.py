import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI image sequences.
    Expected structure: root_dir/<sequence_id>/image_02/data/*.png
    """
    def __init__(self, root_dir: str, sequence_length: int = 20, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset (e.g., data/kitti).
            sequence_length (int): Number of frames per sequence.
            transform (callable, optional): Transform to apply to images.
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Find all image files
        # Assuming structure: root/00/image_02/data/*.png
        self.image_files = sorted(glob.glob(os.path.join(root_dir, "*", "image_02", "data", "*.png")))
        
        if not self.image_files:
            print(f"Warning: No images found in {root_dir}. Make sure structure is root/seq/image_02/data/")
            
        # We can create sequences starting from index 0 to len - seq_len
        self.num_sequences = max(0, len(self.image_files) - sequence_length + 1)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Get sequence of file paths
        seq_paths = self.image_files[idx : idx + self.sequence_length]
        
        images = []
        for p in seq_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        # Stack into (seq_len, C, H, W)
        return torch.stack(images)
