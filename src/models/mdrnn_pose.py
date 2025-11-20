import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from src.data.loaders import KITTIDataset
from src.utils.device import get_compute_device


class DreamerMDRNN(nn.Module):
    def __init__(
        self, latent_dim: int = 32, hidden_size: int = 256, num_gaussians: int = 5
    ):
        """
        MDRNN with a Dual Head:
        1. Vision Head: Predicts future latent vectors (Probabilistic)
        2. Pose Head: Predicts vehicle movement (Deterministic)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        # --- 1. The Memory (LSTM) ---
        # Input: z_t (Visual Latent)
        # Output: hidden state h_t (Context)
        self.lstm = nn.LSTM(latent_dim, hidden_size, batch_first=True)

        # --- 2. The Vision Heads (MDN) ---
        # Pi: Mixing Coefficients (Which Gaussian to use?)
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)

        # Mu: Mean of each Gaussian (Where is the center?)
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_dim)

        # Sigma: Spread of each Gaussian (Uncertainty)
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_dim)

        # --- 3. The Pose Head (Odometry) ---
        # Predicts delta movement: [dx, dy, dz, d_roll, d_pitch, d_yaw]
        # This creates the data for your KITTI submission file.
        self.fc_pose = nn.Linear(hidden_size, 6)

    def forward(
        self,
        z: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Forward pass for Training.
        z shape: (Batch, Seq_Len, Latent_Dim)
        """
        batch_size, seq_len, _ = z.size()

        # 1. LSTM Pass
        lstm_out, next_hidden = self.lstm(z, hidden)

        # Flatten to feed into Linear layers
        flat_out = lstm_out.contiguous().view(-1, self.hidden_size)

        # 2. Vision Predictions (MDN)
        pi = self.fc_pi(flat_out)
        pi = F.softmax(pi, dim=1)
        pi = pi.view(batch_size, seq_len, self.num_gaussians)

        mu = self.fc_mu(flat_out)
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        sigma = self.fc_sigma(flat_out)
        sigma = torch.exp(sigma)
        # CRITICAL STABILITY FIX: Clamp sigma to prevent NaN loss on H100
        sigma = torch.clamp(sigma, min=1e-5, max=10.0)
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        # 3. Pose Prediction (Regression)
        pose_out = self.fc_pose(flat_out)
        pose_out = pose_out.view(batch_size, seq_len, 6)

        return pi, mu, sigma, pose_out, next_hidden

    def loss_function(
        self,
        y_true_latent: torch.Tensor,
        y_true_pose: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        pred_pose: torch.Tensor,
        pose_weight: float = 100.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the combined loss: Vision Loss + Pose Loss.

        Args:
            y_true_latent: The ACTUAL next z vector.
            y_true_pose: The ACTUAL delta movement (dx, dy, etc).
            pose_weight: Scaling factor. Pose numbers are tiny (0.01), MDN is big (10.0).
                         We multiply pose loss by 100-1000 to make the optimizer care about it.
        """
        # --- A. Vision Loss (MDN Negative Log Likelihood) ---
        y_true_latent = y_true_latent.unsqueeze(2).expand_as(mu)

        var = sigma**2
        log_scale = torch.log(sigma)
        sqr_diff = (y_true_latent - mu) ** 2

        # Gaussian Probability Density Function
        log_prob = -0.5 * (
            torch.log(2 * torch.tensor(np.pi)) + 2 * log_scale + sqr_diff / var
        )
        log_prob = torch.sum(log_prob, dim=3)  # Sum across latent dims

        # Log-Sum-Exp for numerical stability
        log_pi = torch.log(pi + 1e-8)
        weighted_log_prob = log_pi + log_prob
        log_prob_total = torch.logsumexp(weighted_log_prob, dim=2)

        loss_mdn = -torch.mean(log_prob_total)

        # --- B. Pose Loss (MSE) ---
        loss_pose = F.mse_loss(pred_pose, y_true_pose)

        # --- C. Total Loss ---
        total_loss = loss_mdn + (loss_pose * pose_weight)

        return total_loss, loss_mdn, loss_pose

    def sample_dream(self, z: torch.Tensor, hidden=None, temperature: float = 1.0):
        """
        Optimized "Dreaming" function.
        Returns the NEXT latent state and the PREDICTED movement.
        """
        with torch.no_grad():
            pi, mu, sigma, pose_out, hidden = self.forward(z, hidden)

            # We only care about the last step in the sequence
            pi = pi[:, -1, :]
            mu = mu[:, -1, :, :]
            sigma = sigma[:, -1, :, :]
            pose_out = pose_out[:, -1, :]  # This is your predicted movement

            # 1. Apply Temperature to Probability Mixing
            if temperature != 1.0:
                pi = torch.log(pi) / temperature
                pi = F.softmax(pi, dim=1)

            # 2. Choose Gaussian Index (Vectorized)
            # Shape: (Batch_Size, 1)
            k_indices = torch.multinomial(pi, 1)

            # 3. Gather the specific Mu/Sigma for that index
            k_expanded = k_indices.unsqueeze(-1).expand(-1, -1, self.latent_dim)

            sampled_mu = torch.gather(mu, 1, k_expanded).squeeze(1)
            sampled_sigma = torch.gather(sigma, 1, k_expanded).squeeze(1)

            # 4. Reparameterize (The actual "Dream")
            epsilon = torch.randn_like(sampled_mu)
            z_next = sampled_mu + (sampled_sigma * epsilon * np.sqrt(temperature))

            return z_next.unsqueeze(1), pose_out, hidden


def training(cfg):
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    img_size = cfg["data"]["img_size"]
    batch_size = cfg["training"]["batch_size"]

    transform = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    )
    dataset = KITTIDataset(
        root_dir=cfg["data"]["path"], sequence_length=1, transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = get_compute_device()
    model = DreamerMDRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for batch in dataloader:
        # Data: (Batch, Seq_Len, Latent_Dim)
        z_seq = batch["z_vector"].to(device)
        # Data: (Batch, Seq_Len, 6)
        pose_seq = batch["pose_delta"].to(device)

        # Input is 0..T-1, Target is 1..T
        z_input = z_seq[:, :-1, :]
        z_target = z_seq[:, 1:, :]
        pose_target = pose_seq[:, 1:, :]

        optimizer.zero_grad()

        pi, mu, sigma, pred_pose, _ = model(z_input)

        # Calculate Loss
        # WEIGHT IS CRITICAL: 100.0 makes 0.001 look like 0.1
        loss, l_mdn, l_pose = model.loss_function(
            z_target, pose_target, pi, mu, sigma, pred_pose, pose_weight=100.0
        )

        loss.backward()

        # Clip gradients (Standard RNN practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
