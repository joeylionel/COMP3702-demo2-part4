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



# ----------------------------------------------
# Task 2: UNet Architecture and Dice Loss
# ----------------------------------------------

class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) * 2 """
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Classic UNet Architecture for OASIS MRI Brain Segmentation.
    Input:  [Batch, 1, H, W]
    Output: [Batch, num_classes, H, W] (Categorical / One-Hot logits)
    """
    def __init__(self, in_channels=1, num_classes=4):
        super(UNet, self).__init__()
        
        # Contracting Path for feature extraction
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Bottleneck 
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Expansive Path 
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)

        # Output Layer (1x1 Convolution to map to num_classes)
        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # encoding downsampling path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)

        # decoding upsampling path with skip connections
        u1 = self.up1(x5)
        d1 = self.conv_up1(torch.cat([u1, x4], dim=1))

        u2 = self.up2(d1)
        d2 = self.conv_up2(torch.cat([u2, x3], dim=1))

        u3 = self.up3(d2)
        d3 = self.conv_up3(torch.cat([u3, x2], dim=1))

        u4 = self.up4(d3)
        d4 = self.conv_up4(torch.cat([u4, x1], dim=1))

        logits = self.outc(d4)
        return logits


class DiceCELoss(nn.Module):
    """
    Binary Cross Entropy + Soft Dice Loss ( One-Hot Categorical Segmentation )
    """
    def __init__(self, smooth=1e-5):
        super(DiceCELoss, self).__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target_onehot, target_indices):
        ce_loss = self.ce(logits, target_indices)

        probs = torch.softmax(logits, dim=1)
        intersection = torch.sum(probs * target_onehot, dim=(2, 3))
        cardinality = torch.sum(probs + target_onehot, dim=(2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)

        return ce_loss + dice_loss


def calculate_dsc(pred_logits, target_indices, num_classes=4):
    """
    calculate Dice Similarity Coefficient (DSC) for independent classes.
    """
    pred_indices = torch.argmax(pred_logits, dim=1)
    dsc_list = []
    
    for c in range(num_classes):
        pred_c = (pred_indices == c).float()
        target_c = (target_indices == c).float()
        
        intersection = torch.sum(pred_c * target_c)
        denom = torch.sum(pred_c) + torch.sum(target_c)
        if denom == 0:
            dsc = 1.0
        else:
            dsc = (2.0 * intersection / denom).item()
        dsc_list.append(dsc)
    return dsc_list