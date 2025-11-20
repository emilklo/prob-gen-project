import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from config.logging import get_logger

logger = get_logger(__name__)


class MDNRNN(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_size: int = 256, num_gaussians: int = 5) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians

        # 1. The LSTM (Memory)
        # Input: z_t (latent_dim)
        # Output: hidden state (hidden_size)
        self.lstm = nn.LSTM(latent_dim, hidden_size, batch_first=True)

        # 2. The MDN Heads (Prediction)
        # We need to predict Pi, Mu, and Sigma from the LSTM output

        # Pi: The mixing coefficients (probability of each Gaussian)
        # Output size: num_gaussians
        self.fc_pi = nn.Linear(hidden_size, num_gaussians)

        # Mu: The means of the Gaussians
        # Output size: num_gaussians * latent_dim
        self.fc_mu = nn.Linear(hidden_size, num_gaussians * latent_dim)

        # Sigma: The standard deviations
        # Output size: num_gaussians * latent_dim
        self.fc_sigma = nn.Linear(hidden_size, num_gaussians * latent_dim)

    def forward(self, z: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        z: Input latent vector (Batch, Seq_Len, Latent_Dim)
        hidden: Previous LSTM state (h_n, c_n)
        """
        # 1. LSTM Pass
        # output shape: (Batch, Seq_Len, Hidden_Size)
        lstm_out, next_hidden = self.lstm(z, hidden)

        # Flatten for linear layers (Batch * Seq, Hidden)
        batch_size, seq_len, _ = lstm_out.size()
        flat_out = lstm_out.contiguous().view(-1, self.hidden_size)

        # 2. Predict Parameters

        # Pi -> Softmax (Probabilities must sum to 1)
        pi = self.fc_pi(flat_out)
        pi = F.softmax(pi, dim=1)
        pi = pi.view(batch_size, seq_len, self.num_gaussians)

        # Mu -> No activation (can be negative)
        mu = self.fc_mu(flat_out)
        mu = mu.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        # Sigma -> Exp (Must be positive)
        sigma = self.fc_sigma(flat_out)
        sigma = torch.exp(sigma)
        sigma = sigma.view(batch_size, seq_len, self.num_gaussians, self.latent_dim)

        return pi, mu, sigma, next_hidden

    def loss_function(self, y_true: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculates Negative Log Likelihood (NLL).
        We want to MAXIMIZE the likelihood that y_true belongs to the distribution.
        So we MINIMIZE the negative log likelihood.

        y_true: The actual next z_t+1 (Batch, Seq, Latent)
        """
        # Expand y_true to match GMM shape: (Batch, Seq, Gaussians, Latent)
        y_true = y_true.unsqueeze(2).expand_as(mu)

        # Calculate Probability density of y_true under each Gaussian
        # Using the formula for Normal Distribution PDF
        var = sigma**2
        log_scale = torch.log(sigma)
        # (x - mu)^2 / (2*sigma^2)
        sqr_diff = (y_true - mu) ** 2
        log_prob = -0.5 * (
            torch.log(2 * torch.tensor(np.pi)) + 2 * log_scale + sqr_diff / var
        )

        # Sum probabilities across Latent Dimensions (assuming diagonal covariance)
        # Shape: (Batch, Seq, Gaussians)
        log_prob = torch.sum(log_prob, dim=3)

        # Weigh by mixing coefficients (Pi)
        # We use Log-Sum-Exp trick for numerical stability
        # log( sum( pi * N(x) ) )
        log_pi = torch.log(pi + 1e-8)  # Add epsilon to prevent log(0)
        weighted_log_prob = log_pi + log_prob

        # Sum across Gaussians
        log_prob_total = torch.logsumexp(weighted_log_prob, dim=2)

        # Mean negative log likelihood
        return -torch.mean(log_prob_total)

    def sample(self, z: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, temperature: float = 1.0) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predicts the NEXT single step z_{t+1} based on z_t.
        This is used for "Dreaming".

        Temperature: Controls diversity.
        High temp (>1) = Random/Creative. Low temp (<1) = Deterministic.
        """
        with torch.no_grad():
            pi, mu, sigma, hidden = self.forward(z, hidden)

            # We are only looking at the last time step
            pi = pi[:, -1, :]
            mu = mu[:, -1, :, :]
            sigma = sigma[:, -1, :, :]

            # 1. Pick which Gaussian to use (based on Pi)
            # Apply temperature to Pi
            pi = torch.log(pi) / temperature
            pi = F.softmax(pi, dim=1)

            # Randomly select index based on probabilities
            m = torch.distributions.Categorical(pi)
            idx = m.sample()  # Shape: (Batch,)

            # 2. Sample from that Gaussian
            # We need to gather the specific Mu/Sigma for the chosen indices
            # This indexing is tricky in PyTorch, so we loop for clarity (batch size is usually 1 during dreaming)
            next_z = []
            for i in range(z.size(0)):  # For each item in batch
                k = idx[i]
                sampled_mu = mu[i, k]
                sampled_sigma = sigma[i, k]

                # Sample: mu + sigma * epsilon * temperature
                epsilon = torch.randn_like(sampled_sigma)
                z_next = sampled_mu + sampled_sigma * epsilon * np.sqrt(temperature)
                next_z.append(z_next)

            return torch.stack(next_z).unsqueeze(1), hidden
