"""
dataset.py
Data loading and preprocessing pipeline for OASIS MRI Brain Dataset.
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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