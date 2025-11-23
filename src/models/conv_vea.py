import torch
import torch.nn as nn

from config.config import LATENT_DIM


class ConvVAE(nn.Module):
    """
    Variational Autoencoder for spatial compression.
    Input: (B, 3, H, W)
    Latent: (B, latent_dim)
    Supports both square and rectangular images.
    """

    def __init__(
        self, latent_dim=LATENT_DIM, img_size=None, img_height=None, img_width=None
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Support both square (img_size) and rectangular (img_height, img_width)
        if img_size is not None:
            self.img_height = img_size
            self.img_width = img_size
        elif img_height is not None and img_width is not None:
            self.img_height = img_height
            self.img_width = img_width
        else:
            raise ValueError(
                "Must provide either img_size or both img_height and img_width"
            )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate feature map size after 4 downsampling layers (factor of 16)
        self.h_feat = self.img_height // 16
        self.w_feat = self.img_width // 16

        if self.h_feat < 1 or self.w_feat < 1:
            raise ValueError(
                f"Image size ({self.img_height}x{self.img_width}) is too small for 4 layers of downsampling."
            )

        self.flatten_dim = 256 * self.h_feat * self.w_feat

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        # RNN input size (for concatenating current and previous latent)
        self.rnn_input_size = self.latent_dim * 2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Helper to get just the latent vector (mu) for the RNN."""
        x_enc = self.encoder(x)
        x_flat = x_enc.view(x_enc.size(0), -1)
        mu = self.fc_mu(x_flat)
        return mu

    def decode(self, z):
        """Decode latent vector back to image."""
        z_dec = self.decoder_input(z)
        z_reshaped = z_dec.view(-1, 256, self.h_feat, self.w_feat)
        recon_x = self.decoder(z_reshaped)
        return recon_x

    def forward(self, x):
        # Encode
        x_enc = self.encoder(x)
        x_flat = x_enc.view(x_enc.size(0), -1)

        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        recon_x = self.decode(z)

        return recon_x, mu, logvar
