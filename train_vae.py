"""
train_vae.py
Training pipeline and Manifold Visualisation for Task 1 (ConvVAE).
COMP3710 Demonstration 2 - Part 4.
"""

import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from dataset import get_vae_dataloader
from modules import ConvVAE, vae_loss_function


def train_vae(data_dir, epochs=15, batch_size=64, lr=1e-3, latent_dim=2, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"--> Training ConvVAE on device: {device}")

    # prepare data loader and model
    train_loader = get_vae_dataloader(data_dir=data_dir, batch_size=batch_size, img_size=(128, 128), shuffle=True)
    model = ConvVAE(in_channels=1, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_bce = 0.0
        total_kl = 0.0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss, bce, kl = vae_loss_function(recon_batch, data, mu, logvar, beta=1.0)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bce += bce.item()
            total_kl += kl.item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_bce = total_bce / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)
        loss_history.append(avg_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Total Loss: {avg_loss:.2f} (BCE: {avg_bce:.2f}, KL: {avg_kl:.2f})")

    # save trained model weights(in the directory "models")
    os.makedirs("models", exist_ok=True)
    weight_path = "models/vae_oasis.pth"
    torch.save(model.state_dict(), weight_path)
    print(f"--> Trained weights saved to: {weight_path}")

    return model, loss_history, device


def plot_latent_manifold(model, device, n=12, digit_size=128, save_path="vae_manifold.png"):
    """
    Manifold Visualisation:
    Generates a grid of images by sampling the latent space and decoding them.
    """
    model.eval()
    figure = np.zeros((digit_size * n, digit_size * n))

    # generate grid in range of [-2.5, 2.5] in latent space 
    grid_x = np.linspace(-2.5, 2.5, n)
    grid_y = np.linspace(-2.5, 2.5, n)

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float32).to(device)
                
                # Decode the latent vector to reconstruct the image
                d = model.decoder_input(z_sample)
                d = d.view(-1, 256, 8, 8)
                x_decoded = model.decoder(d)
                
                digit = x_decoded[0][0].cpu().numpy()
                figure[i * digit_size : (i + 1) * digit_size,
                       j * digit_size : (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10), facecolor='white')
    plt.imshow(figure, cmap='gray')
    plt.title("OASIS Brain MRI 2D Latent Space Manifold", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    print(f"--> Latent space manifold plot saved as: {save_path}")
    plt.show()


if __name__ == "__main__":
    import sys
    # default data path for local testing
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/keras_png_slices_data/keras_png_slices_train"
    
    # Train the VAE model and visualize the latent space manifold
    trained_model, history, dev = train_vae(data_dir=data_path, epochs=10, batch_size=64, lr=1e-3)
    
    # Plot the latent space manifold after training
    plot_latent_manifold(trained_model, dev, n=10, save_path="vae_manifold.png")