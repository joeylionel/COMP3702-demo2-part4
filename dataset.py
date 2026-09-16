"""
dataset.py
Data loading and preprocessing pipeline for OASIS MRI Brain Dataset.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

class OASISVAEDataset(Dataset):
    """
    Dataset loader for Task 1: Variational Autoencoder (VAE).
    Loads 2D MRI brain PNG slices and normalizes them to [0, 1].
    """
    def __init__(self, data_dir, img_size=(128, 128)):
        super(OASISVAEDataset, self).__init__()
        self.data_dir = Path(data_dir)
        
        # Scan all images in the directory (compatible with .png suffix)
        self.image_paths = sorted(list(self.data_dir.rglob("*.png")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No PNG images found in directory: {self.data_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # Convert to a Tensor and automatically normalize to [0.0, 1.0].
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        return self.transform(img)


def get_vae_dataloader(data_dir, batch_size=32, img_size=(128, 128), shuffle=True):
    dataset = OASISVAEDataset(data_dir=data_dir, img_size=img_size)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, 
        pin_memory=True
    )
    return loader


if __name__ == "__main__":
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    try:
        loader = get_vae_dataloader(test_dir, batch_size=4)
        sample_batch = next(iter(loader))
        print(f"[Dataset Test Pass] Total PNG slices: {len(loader.dataset)}")
        print(f"Batch Tensor Shape: {sample_batch.shape}, Value Range: [{sample_batch.min():.2f}, {sample_batch.max():.2f}]")
    except Exception as e:
        print(f"[Dataset Test Failed]: {e}")



# ---------------------------------------------------
# Task 2: UNet Segmentation Dataset
# ---------------------------------------------------

class OASISSegDataset(Dataset):
    """
    Dataset loader for Task 2: UNet Brain MRI Segmentation.
    Loads paired MRI slices and ground-truth segmentation masks.
    Converts masks into Categorical (One-Hot) format.
    """
    def __init__(self, img_dir, mask_dir, img_size=(128, 128),num_classes=4):
        super(OASISSegDataset, self).__init__()
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.num_classes = num_classes

        # Recursively retrieve all PNG files sorted for alignment
        self.img_paths = sorted(list(self.img_dir.rglob("*.png")))
        self.mask_paths = sorted(list(self.mask_dir.rglob("*.png")))

        # Sanity checks
        if len(self.img_paths) == 0 or len(self.mask_paths) == 0:
            raise RuntimeError(
                f"Data not found: {len(self.img_paths)} images, {len(self.mask_paths)} masks."
            )
        assert len(self.img_paths) == len(self.mask_paths), (
            f"Mismatch between images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) count!"
        )

        # NEAREST interpolation avoids introducing artificial class values at boundaries
        self.resize = transforms.Resize(
            img_size, interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. Load MRI slice in single-channel grayscale and convert to [1, H, W] float Tensor
        img = Image.open(self.img_paths[idx]).convert("L")
        img = self.resize(img)
        img_tensor = transforms.ToTensor()(img)  # Shape: [1, H, W], Range: [0.0, 1.0]

        # 2. Load mask and resize with nearest-neighbor interpolation
        mask = Image.open(self.mask_paths[idx])
        mask = self.resize(mask)
        mask_np = np.array(mask, dtype=np.int64)

        # Map OASIS intensity levels [0, 85, 170, 255] to categorical IDs [0, 1, 2, 3] if needed
        if mask_np.max() > 4:
            mask_indices = np.digitize(mask_np, bins=[42, 127, 212])
        else:
            mask_indices = mask_np

        mask_tensor = torch.from_numpy(mask_indices).long()  # Shape: [H, W]

        # 3. One-Hot encoding: [H, W] -> [H, W, C] -> [C, H, W]
        mask_onehot = F.one_hot(mask_tensor, num_classes=self.num_classes)
        mask_onehot = mask_onehot.permute(2, 0, 1).float()  # Channel-first float Tensor

        return img_tensor, mask_onehot, mask_tensor


    def get_seg_dataloaders(base_dir, batch_size=16, img_size=(128, 128)):
        """Create DataLoader instances for training, validation, and test splits."""
        base = Path(base_dir)

        train_ds = OASISSegDataset(
            img_dir=base / "keras_png_slices_train",
            mask_dir=base / "keras_png_slices_seg_train",
            img_size=img_size,
        )
        val_ds = OASISSegDataset(
            img_dir=base / "keras_png_slices_validate",
            mask_dir=base / "keras_png_slices_seg_validate",
            img_size=img_size,
        )
        test_ds = OASISSegDataset(
            img_dir=base / "keras_png_slices_test",
            mask_dir=base / "keras_png_slices_seg_test",
            img_size=img_size,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader