"""
dataset.py  —  Newa Script OCR
Loads images from your dataset_final/ folder into PyTorch tensors.

Your folder structure must look like:
  dataset_final/
    train/
      class_0/   (e.g. "vowel_a")
        img1.png
        img2.png
        ...
      class_1/
        ...
    val/
      class_0/
        ...
    test/
      class_0/
        ...

If your folders are named differently (e.g. numbers 0-66),
that also works — PyTorch reads whatever folder names you have.
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Settings ─────────────────────────────────────────────────────────────────

DATASET_DIR = "dataset_final"   # change this if your folder has a different name
IMAGE_SIZE  = 128               # your images are 128x128
BATCH_SIZE  = 64                # how many images to process at once
                                # 64 is good for a mid-range GPU
                                # reduce to 32 if you get "out of memory" errors
NUM_WORKERS = 2                 # parallel loading threads (2-4 is fine on Windows)
NUM_CLASSES = 67                # your 67 character classes

# ── Transforms (what to do to each image before feeding to CNN) ───────────────

# Training transform: small random augmentations make the model more robust
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # ensure single channel
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(degrees=10),          # random slight rotation
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # small shifts
    transforms.ToTensor(),                          # converts pixel 0-255 to 0.0-1.0
    transforms.Normalize(mean=[0.5], std=[0.5]),    # normalize to -1 to +1 range
])

# Validation/Test transform: no augmentation — just resize and normalize
eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ── Load datasets ─────────────────────────────────────────────────────────────

def get_dataloaders(dataset_dir=DATASET_DIR):
    """
    Returns three DataLoaders: train, val, test.
    Also returns the list of class names.
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir   = os.path.join(dataset_dir, "val")
    test_dir  = os.path.join(dataset_dir, "test")

    # ImageFolder automatically reads subfolder names as class labels
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    print(f"Classes found : {len(train_dataset.classes)}")
    print(f"Train images  : {len(train_dataset)}")
    print(f"Val images    : {len(val_dataset)}")
    print(f"Test images   : {len(test_dataset)}")

    # Sanity check
    assert len(train_dataset.classes) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes but found {len(train_dataset.classes)}. "
        f"Check your dataset folder."
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,           # shuffle training data every epoch
        num_workers=NUM_WORKERS,
        pin_memory=True,        # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # no shuffle for evaluation
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


# ── Quick test (run this file directly to check everything loads) ─────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # Grab one batch and show its shape
    images, labels = next(iter(train_loader))
    print(f"\nOne batch shape : {images.shape}")   # should be [64, 1, 128, 128]
    print(f"Label range     : {labels.min()} to {labels.max()}")
    print(f"\nFirst 5 class names: {class_names[:5]}")
    print("\ndataset.py  OK — your dataset loads correctly.")