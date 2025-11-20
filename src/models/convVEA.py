import torch
import torch.nn as nn


class VAE(nn.Module):
    """
    Variational Autoencoder for spatial compression.
    Input: (B, 3, Height, Width)
    Latent: (B, latent_dim)
    """

    def __init__(self, latent_dim=128, img_height=128, img_width=416):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width

        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: Input -> 32 channels
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Layer 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Layer 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Layer 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate feature map size after 4 downsampling layers (factor of 16)
        self.h_feat = img_height // 16
        self.w_feat = img_width // 16

        if self.h_feat < 1 or self.w_feat < 1:
            raise ValueError(
                f"Image size ({img_height}x{img_width}) is too small for 4 layers of downsampling."
            )

        self.flatten_dim = 256 * self.h_feat * self.w_feat

        # Latent vectors
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # Decoder Setup
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Use Sigmoid if input images are normalized to [0, 1]
        )

        # --- YOUR REQUESTED INFO ---
        # Calculate the exact input size for the RNN (Option A: Concatenation)
        self.rnn_input_size = self.latent_dim * 2
        print(f"[-] VAE Initialized.")
        print(f"    Input Image: {img_height}x{img_width}")
        print(f"    Latent Dim:  {latent_dim}")
        print(
            f"    RNN Input (Option A): {self.rnn_input_size} (concatenating t and t-1)"
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        """Helper to just get the latent vector (mu) for the RNN"""
        x_enc = self.encoder(x)
        x_flat = x_enc.view(x_enc.size(0), -1)
        mu = self.fc_mu(x_flat)
        return mu

    def forward(self, x):
        # Encode
        x_enc = self.encoder(x)
        x_flat = x_enc.view(x_enc.size(0), -1)

        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode
        z_dec = self.decoder_input(z)
        z_reshaped = z_dec.view(-1, 256, self.h_feat, self.w_feat)
        recon_x = self.decoder(z_reshaped)

        return recon_x, mu, logvar


# --- Example Usage for your RNN ---
vae_model = VAE(latent_dim=128, img_height=128, img_width=416)

# Later in your RNN definition, you can do this:
# rnn = nn.LSTM(input_size=vae_model.rnn_input_size, ...)
