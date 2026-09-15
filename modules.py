"""
modules.py
Convolutional Neural Network
Task 1: Convolutional Variational Autoencoder (VAE).
"""

import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder for OASIS Brain MRI slices.
    Input image dimension: [Batch, 1, 128, 128]
    Latent vector dimension: latent_dim (default 2 for direct 2D manifold visualisation)
    """
    def __init__(self, in_channels=1, latent_dim=2):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim

        # -----------------------------------------------------------
        # Encoder: extract spatial features under convolutional layers
        # [1, 128, 128] -> [32, 64, 64] -> [64, 32, 32] -> [128, 16, 16] -> [256, 8, 8]
        # -----------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.flatten_dim = 256 * 8 * 8

        # Mapping to latent space mean (mu) and log variance (log_var)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # -----------------------------------------------------------
        # Decoder: transpose convolutional layers for upsampling
        # [latent_dim] -> [256, 8, 8] -> [128, 16, 16] -> [64, 32, 32] -> [32, 64, 64] -> [1, 128, 128]
        # -----------------------------------------------------------
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # reconstruction output in [0, 1] range
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick:
        z = mu + sigma * epsilon, where epsilon ~ N(0, I)
        This allows the gradient to flow through the sampling process during backpropagation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)

        d = self.decoder_input(z)
        d = d.view(-1, 256, 8, 8)
        reconstruction = self.decoder(d)
        return reconstruction, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE ELBO Combined Loss Function:
    Loss = Reconstruction Loss (BCE) + beta * KL Divergence
    """
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Module Forward Inference and Loss Calculation Self-Check
    sample_input = torch.randn(4, 1, 128, 128)
    model = ConvVAE(in_channels=1, latent_dim=2)
    recon, mu, logvar = model(sample_input)
    loss, bce, kl = vae_loss_function(recon, torch.sigmoid(sample_input), mu, logvar)

    print("[modules.py Test Passed]")
    print(f"Input shape:          {sample_input.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent vector shape:  {mu.shape}")
    print(f"Calculated Loss:      {loss.item():.2f} (BCE: {bce.item():.2f}, KL: {kl.item():.2f})")