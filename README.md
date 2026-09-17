# COMP3710 Part 4 
**Student Name:** YIZHUO WU
**Student ID:** 49732823 
**Task Difficulty Level:** Medium (Task 1: VAE + Task 2: UNet Segmentation)

## 1. Project Overview
This repository contains the implementation of deep learning architectures for brain MRI analysis using the preprocessed OASIS dataset:
- **Task 1 (VAE):** A Convolutional Variational Autoencoder trained to learn the continuous latent representation of brain MRI scans and reconstruct 2D latent space manifolds.
- **Task 2 (UNet):** A 2D UNet segmentation network featuring skip connections and categorical (one-hot) multi-class outputs to segment brain tissues into 4 anatomical structures.

## 2. Dependencies and Environment
- Python 3.10+
- PyTorch >= 2.0.0
- Torchvision
- Pillow
- NumPy
- Matplotlib
- scikit-learn

Install all required packages via:
```bash
pip install torch torchvision pillow numpy matplotlib scikit-learn
