import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from config.logging import get_logger
from data.odometry import KITTIOdometryDataset
from models.mdrnn_pose import DreamerMDRNN
from models.world_model import ConvVAE
from utils.device import get_compute_device


logger = get_logger(__name__)


def training(cfg):
    # 1. Setup Configuration
    img_height = 128
    img_width = 416
    seq_len = 10  # CRITICAL: RNNs need a sequence, not just 1 frame!
    batch_size = 8  # Keep small for video data

    # 2. Transforms
    transform = transforms.Compose(
        [transforms.Resize((img_height, img_width)), transforms.ToTensor()]
    )

    # 3. Initialize Dataset (Point to your download locations)
    dataset = KITTIOdometryDataset(
        data_dir="dataset/sequences/00/image_2",
        pose_dir="dataset/poses",
        sequence_id="00",
        seq_len=seq_len,
        transform=transform,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # 4. Models
    device = get_compute_device()

    # You need the VAE to encode images into Z
    vae = ConvVAE(latent_dim=128, img_height=img_height, img_width=img_width).to(device)

    # Your RNN takes (Z_t, Z_t-1), so input size is latent*2
    rnn = DreamerMDRNN(latent_dim=128, hidden_size=512).to(device)

    optimizer = torch.optim.Adam(
        list(vae.parameters()) + list(rnn.parameters()), lr=1e-4
    )

    # 5. Loop
    for batch_idx, (images, pose_deltas) in enumerate(dataloader):
        # images: (B, Seq, 3, H, W)
        # pose_deltas: (B, Seq, 6)

        images = images.to(device)
        pose_targets = pose_deltas.to(device)

        # --- A. Get Latents (Z) from VAE ---
        b, s, c, h, w = images.size()

        # Flatten batch and sequence to pass through VAE
        images_flat = images.view(b * s, c, h, w)

        # Get Z (Mean) only - we don't need reconstruction for the RNN part
        z_flat = vae.encode(images_flat)

        # Reshape back to (Batch, Sequence, Latent)
        z_sequence = z_flat.view(b, s, -1)

        # --- B. Prepare RNN Inputs ---
        # Input: Frames 0 to 9
        # Target Pose: Frames 1 to 10 (Movement from 0->1, 1->2, etc)

        # We need to pair z_t and z_t-1 for the RNN
        # Or, if your RNN handles the memory, just feed the sequence.
        # Based on your code:

        rnn_input = z_sequence[:, :-1, :]  # Input Z

        # For the vision loss, we predict the NEXT Z
        z_target = z_sequence[:, 1:, :]  # Target Z

        # For pose loss, we match the deltas
        pose_target = pose_targets[:, 1:, :]

        # --- C. Forward & Loss ---
        optimizer.zero_grad()

        pi, mu, sigma, pred_pose, _ = rnn(rnn_input)

        loss, l_mdn, l_pose = rnn.loss_function(
            z_target, pose_target, pi, mu, sigma, pred_pose, pose_weight=100.0
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.0)
        optimizer.step()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx} | Total: {loss.item():.4f} | Pose: {l_pose.item():.4f}"
            )
