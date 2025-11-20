import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Corrected imports based on file structure
from config.logging import get_logger
from src.data.odometry_loader import KITTIOdometryDataset
from src.models.mdnrnn_pose import DreamerMDRNN
from src.models.world_model import ConvVAE
from src.utils.device import get_config, Config


logger = get_logger(__name__)


def training(cfg: Config):
    # --- 1. Setup Configuration ---
    # Use config values instead of hardcoded ones
    img_height = cfg.data.img_height
    img_width = cfg.data.img_width

    seq_len = cfg.data.rnn_sequence_length
    batch_size = cfg.training.batch_size
    device = torch.device(cfg.device)

    print(
        f"[-] Configuration loaded: Device={device}, Batch={batch_size}, SeqLen={seq_len}"
    )

    # --- 2. Transforms (CRITICAL FIX) ---
    # You must resize images to what the VAE expects
    transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ]
    )

    # --- 3. Initialize Dataset ---
    # ensure root_dir points to the folder containing '00', '01', etc.
    # cfg.data.path is "data/kitti"
    dataset = KITTIOdometryDataset(
        root_dir=cfg.data.path,  # e.g., /path/to/dataset/sequences
        pose_dir="dataset/poses",  # e.g., /path/to/dataset/poses (This might need to be in config too?)
        seq_len=seq_len,
        transform=transform,  # <--- Added this!
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4
    )

    # --- 4. Models ---

    # A. Load VAE and FREEZE it (Recommended)
    vae = ConvVAE(
        latent_dim=cfg.model.vae.latent_dim, img_height=img_height, img_width=img_width
    ).to(device)

    # Load weights from your previous VAE training run
    # We should probably look for the best checkpoint or a specific one
    # For now, let's assume it's in the default output location or specified
    # Let's try to find a reasonable default path based on run_name if possible,
    # but since VAE and RNN might be trained separately, we might need a specific arg for VAE path.
    # For now, I'll keep the hardcoded path but make it relative to project root if needed.
    vae_path = "checkpoints/vae_final.pth"  # <--- UPDATE THIS PATH

    if os.path.exists(vae_path):
        print(f"[-] Loading VAE weights from {vae_path}")
        checkpoint = torch.load(vae_path, map_location=device)
        vae.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(
            "[!] WARNING: No Pre-trained VAE found. Training from scratch (Unstable!)"
        )

    # Freeze VAE (We only want to train the RNN now)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # B. Initialize RNN
    # Note: Input size is latent_dim because you are feeding z_t
    rnn = DreamerMDRNN(
        latent_dim=cfg.model.vae.latent_dim, hidden_size=cfg.model.rnn.hidden_size
    ).to(device)

    # Optimize ONLY the RNN parameters
    optimizer = torch.optim.Adam(rnn.parameters(), lr=cfg.training.learning_rate)

    print(f"[-] Starting RNN Training on {len(dataset)} sequences...")

    # --- 5. Loop ---
    for batch_idx, (images, pose_deltas) in enumerate(dataloader):
        images = images.to(device)  # (B, Seq, 3, H, W)
        pose_targets = pose_deltas.to(device)  # (B, Seq, 6)

        # --- A. Get Latents (Z) from VAE ---
        # We use torch.no_grad() because VAE is frozen
        with torch.no_grad():
            b, s, c, h, w = images.size()
            images_flat = images.view(b * s, c, h, w)

            # Use the VAE to get the feature vector (Mean)
            z_flat = vae.encode(images_flat)

            # Reshape back to sequence
            z_sequence = z_flat.view(b, s, -1)  # (B, Seq, 128)

        # --- B. Prepare RNN Inputs ---
        # Input:  z_0, z_1, ... z_8
        # Target: z_1, z_2, ... z_9
        rnn_input = z_sequence[:, :-1, :]
        z_target = z_sequence[:, 1:, :]

        # Pose Target: Delta_0->1, Delta_1->2 ...
        # The dataset returns deltas matching the image index,
        # so index 1 is the move from 0 to 1.
        pose_target_slice = pose_targets[:, 1:, :]

        # --- C. Forward & Loss ---
        optimizer.zero_grad()

        # Pass the sequence to RNN
        # pi, mu, sigma, pred_pose, _ = rnn(rnn_input)
        # Note: Your RNN returns 5 values, ensure unpacking is correct
        pi, mu, sigma, pred_pose, _ = rnn(rnn_input)

        loss, l_mdn, l_pose = rnn.loss_function(
            z_target,
            pose_target_slice,
            pi,
            mu,
            sigma,
            pred_pose,
            pose_weight=1000.0,  # Increase this if Pose prediction is bad
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx} | Total: {loss.item():.4f} | "
                f"MDN: {l_mdn.item():.4f} | Pose: {l_pose.item():.4f}"
            )


if __name__ == "__main__":
    cfg = get_config()
    training(cfg)
