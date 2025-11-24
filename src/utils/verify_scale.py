import torch
from torch.utils.data import DataLoader
from src.data.odometry_loader import KITTIOdometryDataset
from src.utils.device import get_config
import torchvision.transforms as transforms
import numpy as np


def verify_scale():
    cfg = get_config()

    # Initialize dataset
    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,
        seq_len=cfg.rnn.sequence_length,
        pose_dir=cfg.data.pose_path,
        test_sequences=cfg.data.test_sequences,
        transform=transforms.ToTensor(),
    )

    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4)

    print("[-] Collecting stats...")
    all_deltas = []

    # Collect a subset of data (e.g., first 10 batches) to save time, or all if fast enough
    for i, (_, deltas) in enumerate(dataloader):
        all_deltas.append(deltas)
        if i >= 20:  # Check 2000 samples
            break

    all_deltas = torch.cat(all_deltas, dim=0)  # (N, Seq, 6)
    all_deltas = all_deltas.view(-1, 6)  # Flatten sequence

    # Translation: indices 0, 1, 2
    trans = all_deltas[:, :3]
    # Rotation: indices 3, 4, 5
    rot = all_deltas[:, 3:]

    print("\n[Translation Stats (Meters)]")
    print(f"  Mean (Abs): {trans.abs().mean().item():.6f}")
    print(f"  Max:        {trans.max().item():.6f}")
    print(f"  Min:        {trans.min().item():.6f}")
    print(f"  Std:        {trans.std().item():.6f}")

    print("\n[Rotation Stats (Radians)]")
    print(f"  Mean (Abs): {rot.abs().mean().item():.6f}")
    print(f"  Max:        {rot.max().item():.6f}")
    print(f"  Min:        {rot.min().item():.6f}")
    print(f"  Std:        {rot.std().item():.6f}")

    ratio = trans.abs().mean().item() / (rot.abs().mean().item() + 1e-9)
    print(f"\n[Scale Ratio]")
    print(f"  Translation / Rotation ~ {ratio:.2f}")


if __name__ == "__main__":
    verify_scale()
