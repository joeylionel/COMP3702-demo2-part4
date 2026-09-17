import os
from py_compile import main
from py_compile import main
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from dataset import get_seg_dataloaders
from modules import UNet, DiceCELoss, calculate_dsc


def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch over the dataset.

    Returns:
        float: Average training loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for imgs, masks_onehot, masks_indices in dataloader:
        imgs = imgs.to(device)
        masks_onehot = masks_onehot.to(device)
        masks_indices = masks_indices.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks_onehot, masks_indices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device, num_classes=4):
    """
    Evaluate the model on validation/test data.

    Returns:
        tuple: (avg_loss, mean_dsc_per_class)
            - avg_loss (float): Average loss across all batches.
            - mean_dsc_per_class (list of float): Mean DSC for each class.
    """
    model.eval()
    total_loss = 0.0
    all_dsc = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for imgs, masks_onehot, masks_indices in dataloader:
            imgs = imgs.to(device)
            masks_onehot = masks_onehot.to(device)
            masks_indices = masks_indices.to(device)

            logits = model(imgs)
            loss = criterion(logits, masks_onehot, masks_indices)
            total_loss += loss.item()

            dsc_scores = calculate_dsc(logits, masks_indices, num_classes=num_classes)
            for c in range(num_classes):
                all_dsc[c].append(dsc_scores[c])

    avg_loss = total_loss / len(dataloader)
    mean_dsc_per_class = [float(np.mean(all_dsc[c])) for c in range(num_classes)]

    return avg_loss, mean_dsc_per_class

def run_demo_inference(
    model, test_loader, device, save_path="unet_demo_prediction.png"
):
    """
    Runs inference on a single sample from the test loader and displays
    a 1x3 comparison plot between Input MRI, Ground Truth, and Prediction.
    """
    model.eval()

    # 1. Fetch one batch and extract the first sample
    imgs, masks_onehot, masks_indices = next(iter(test_loader))
    img = imgs[0:1].to(device)  # Keep batch dimension: [1, 1, H, W]
    gt_mask = masks_indices[0].cpu().numpy()  # [H, W]

    # 2. Run prediction and extract class labels via argmax
    with torch.no_grad():
        logits = model(img)
        pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

    # 3. Create 1x3 visualization grid
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")

    # Input MRI Slice
    axes[0].imshow(img[0, 0].cpu().numpy(), cmap="gray")
    axes[0].set_title("Input MRI Slice")
    axes[0].axis("off")

    # Ground Truth Mask
    axes[1].imshow(gt_mask, cmap="tab10", vmin=0, vmax=9)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # UNet Predicted Mask
    axes[2].imshow(pred_mask, cmap="tab10", vmin=0, vmax=9)
    axes[2].set_title("UNet Predicted Mask")
    axes[2].axis("off")

    # 4. Save figure and display
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"--> Demo prediction plot saved as: {save_path}")
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 50)
    print(f"--> Training UNet on device: {device}")
    print("--> Target Metric: DSC > 0.9 for all labels")
    print("=" * 50)

    # Hyperparameters and data paths
    data_base = "./data/keras_png_slices_data"
    batch_size = 16
    lr = 5e-4
    epochs = 12
    num_classes = 4
    class_names = ["Background", "CSF", "Gray Matter", "White Matter"]
    best_model_path = "models/unet_oasis_best.pth"

    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_seg_dataloaders(
        base_dir=data_base, batch_size=batch_size, img_size=(128, 128)
    )

    # 2. Instantiate Model, Loss, and Optimizer
    model = UNet(in_channels=1, num_classes=num_classes).to(device)
    criterion = DiceCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_mean_dsc = 0.0

    # 3. Training & Validation Loop
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_dsc = evaluate(
            model, val_loader, criterion, device, num_classes=num_classes
        )

        mean_dsc = float(np.mean(val_dsc))
        print(
            f"\n[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Mean DSC: {mean_dsc:.4f}"
        )
        for c in range(num_classes):
            print(f"    - {class_names[c]:<13}: DSC = {val_dsc[c]:.4f}")

        # Checkpoint the best model by validation mean DSC
        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            torch.save(model.state_dict(), best_model_path)
            print(
                f"    >>> Best model saved! (Mean DSC: {best_mean_dsc:.4f}) to {best_model_path}"
            )

    # 4. Final Evaluation on Test Set using Best Saved Weights
    print("\n" + "=" * 50)
    print("--> Final Evaluation on Test Set using Best Weights")
    print("=" * 50)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_dsc = evaluate(
        model, test_loader, criterion, device, num_classes=num_classes
    )

    all_pass = True
    for c in range(num_classes):
        passed = test_dsc[c] > 0.90
        all_pass = all_pass and passed
        status = "PASSED (>0.9)" if passed else "WARNING (<=0.9)"
        print(f"Test DSC [{class_names[c]}]: {test_dsc[c]:.4f} --> {status}")

    print(f"Overall Mean Test DSC: {np.mean(test_dsc):.4f}")
    if all_pass:
        print("--> SUCCESS: All labels achieved > 0.9 DSC requirement!")
    else:
        print("--> NOTE: One or more tissue types did not meet the > 0.9 target.")

    # 5. Visual Inference Demo
    run_demo_inference(
        model, test_loader, device, save_path="unet_demo_prediction.png"
    )


if __name__ == "__main__":
    main()