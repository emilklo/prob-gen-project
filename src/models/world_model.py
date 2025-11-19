import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """
    Variational Autoencoder for spatial compression.
    Input: (B, 3, H, W)
    Latent: (B, latent_dim)
    """
    def __init__(self, latent_dim=32, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

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
        
        # Calculate size of flattened features
        # 4 layers of stride 2 -> reduction factor of 2^4 = 16
        self.feature_size = img_size // 16
        if self.feature_size < 1:
            raise ValueError(f"Image size {img_size} is too small for 4 layers of downsampling.")
            
        self.flatten_dim = 256 * self.feature_size * self.feature_size
        
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
            nn.Sigmoid() 
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        z_reshaped = z_dec.view(-1, 256, self.feature_size, self.feature_size)
        recon_x = self.decoder(z_reshaped)
        
        return recon_x, mu, logvar

class MDNRNN(nn.Module):
    """
    Mixture Density Network - RNN for temporal prediction.
    """
    def __init__(self, latent_dim=32, hidden_size=256, num_gaussians=5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_gaussians = num_gaussians
        # Define layers here

    def forward(self, z, hidden):
        """
        Predicts next latent state distribution.
        """
        # Placeholder
        return None
