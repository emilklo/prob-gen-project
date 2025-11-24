import torch
import numpy as np
import os
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms

# Project imports
from src.utils.device import get_config
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.conv_vea import ConvVAE

def preprocess_dataset():
    cfg = get_config()
    device = torch.device(cfg.device)
    
    # 1. Output Directory
    save_dir = Path("data/processed_latents")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Saving processed sequences to: {save_dir}")

    # 2. Load VAE (Frozen)
    print("[-] Loading VAE...")
    vae = ConvVAE(
        latent_dim=cfg.vae.latent_dim,
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width
    ).to(device)

    # Point to the actual trained VAE
    vae_path = "outputs/vae_z64_img128_mps2/checkpoints/vae_epoch_20.pth"
    
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"VAE checkpoint not found at {vae_path}")
        
    checkpoint = torch.load(vae_path, map_location=device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()

    # 3. Process Sequences Individually
    # We assume your config has a list of sequences, or we define them here
    all_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    with torch.no_grad():
        for seq_id in all_sequences:
            print(f"    Processing Sequence {seq_id}...")
            
            # Define transform
            tf = transforms.Compose([
                transforms.Resize((cfg.data.img_height, cfg.data.img_width)),
                transforms.ToTensor()
            ])
            
            # Initialize Dataset for SINGLE sequence with seq_len=1
            # We want frame-by-frame encoding, no windows yet.
            try:
                dataset = KITTIOdometryDataset(
                    root_dir=cfg.data.path,
                    pose_dir=cfg.data.pose_path,
                    train_sequences=[seq_id], # Force specific sequence
                    seq_len=1,                # One frame at a time
                    transform=tf              # Pass transform directly
                )
            except Exception as e:
                print(f"    [!] Error loading sequence {seq_id}: {e}")
                continue
            
            if len(dataset) == 0:
                print(f"    [!] Warning: Sequence {seq_id} has 0 samples. Skipping.")
                continue

            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

            mu_list = []
            logvar_list = []
            pose_list = []

            for images, poses in tqdm(loader, desc=f"Seq {seq_id}"):
                # images: (B, 1, C, H, W) -> squeeze to (B, C, H, W)
                # poses:  (B, 1, 6)       -> squeeze to (B, 6)
                images = images.to(device).squeeze(1)
                poses = poses.squeeze(1) # Keep on CPU or move to numpy later

                # VAE Inference
                # Assuming your VAE returns: recon, mu, logvar
                _, mu, logvar = vae(images)

                mu_list.append(mu.cpu().numpy())
                logvar_list.append(logvar.cpu().numpy())
                pose_list.append(poses.numpy())

            if not mu_list:
                print(f"    [!] No data collected for sequence {seq_id}")
                continue

            # Concatenate all batches for this sequence
            mu_arr = np.concatenate(mu_list, axis=0)      # (N_frames, Latent_Dim)
            logvar_arr = np.concatenate(logvar_list, axis=0)
            pose_arr = np.concatenate(pose_list, axis=0)  # (N_frames, 6)

            # Save as .npz
            save_path = save_dir / f"{seq_id}.npz"
            np.savez_compressed(
                save_path,
                mu=mu_arr,
                logvar=logvar_arr,
                pose=pose_arr
            )
            print(f"    [+] Saved {save_path}")

    print("[+] Pre-processing complete.")

if __name__ == "__main__":
    preprocess_dataset()
