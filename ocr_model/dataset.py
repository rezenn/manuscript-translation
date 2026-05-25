"""
dataset.py  —  Newa Script OCR
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
This file teaches PyTorch HOW TO READ your image files.

Your dataset_final/ folder looks like this:
    dataset_final/
        train/
            𑐎/          ← folder name IS the character label
                00001.png
                00002.png
                ...
            𑐏/
                00001.png
                ...
        val/
            𑐎/
                ...
        test/
            𑐎/
                ...

PyTorch needs a "DataLoader" — think of it as a conveyor belt that:
  1. Picks up a batch of images (e.g. 64 at a time)
  2. Converts them to numbers (tensors)
  3. Applies small random changes to make the model more robust
  4. Hands the batch to the model for training

WHAT IS A TENSOR?
─────────────────
A tensor is just a multi-dimensional array of numbers.
One grayscale image of size 64×64 pixels becomes:
    shape [1, 64, 64]
    (channels=1, height=64, width=64)
    each value is a float between -1.0 and +1.0

A batch of 64 such images becomes:
    shape [64, 1, 64, 64]
    (batch=64, channels=1, height=64, width=64)

WHAT IS NORMALISATION?
───────────────────────
Pixel values are 0–255 (integers).
Neural networks learn better when values are small floats near 0.
We normalise: new_value = (pixel/255 - 0.5) / 0.5
This maps:   0   → -1.0
             128 → 0.0
             255 → +1.0
"""

import os
import json
from pathlib import Path

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# SETTINGS — edit these if anything changes
# ═══════════════════════════════════════════════════════════════════

DATASET_DIR = "dataset_final"   # root folder with train/ val/ test/
IMAGE_SIZE  = 64                # resize every image to 64×64 pixels
                                # 64 is enough for character recognition
                                # and trains 4× faster than 128×128
BATCH_SIZE  = 64                # images processed together each step
                                # reduce to 32 if you get memory errors
NUM_WORKERS = 2                 # background threads for loading images
                                # set to 0 on Windows if you get errors


# ═══════════════════════════════════════════════════════════════════
# CLASS MAP  —  which folder name maps to which integer label
# ═══════════════════════════════════════════════════════════════════
#
# The model outputs numbers, not characters. We need a stable mapping:
#   "𑐎" → 0,   "𑐏" → 1,   "𑐐" → 2,  ...  "𑐴" → 81
#
# We save this mapping to class_map.json so:
#   - Training and evaluation always use identical numbering
#   - After training you can look up what number 34 means
#
# ═══════════════════════════════════════════════════════════════════

def build_class_map(dataset_dir, save_path="class_map.json"):
    """
    Scan dataset_final/train/ for subfolder names.
    Return a dict: {class_name_string: integer_index}
    Save it to JSON for later use.
    """
    train_dir = Path(dataset_dir) / "train"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Cannot find: {train_dir}\n"
            f"Make sure you ran build_data.py first."
        )

    # Sort alphabetically for consistency across runs
    class_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    class_map   = {name: idx for idx, name in enumerate(class_names)}

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, ensure_ascii=False, indent=2)

    print(f"Class map built: {len(class_map)} classes → saved to {save_path}")
    return class_map


def load_class_map(path="class_map.json"):
    """Load a previously saved class map from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# TRANSFORMS  —  what to do to each image before feeding the model
# ═══════════════════════════════════════════════════════════════════
#
# Think of transforms as a processing pipeline every image goes through.
#
# TRAINING transform includes AUGMENTATION:
#   Small random changes make the model see each image slightly
#   differently every epoch, which helps it generalise to new data.
#
# EVALUATION transform: NO augmentation.
#   We want a fair, consistent measurement of accuracy.
#
# ═══════════════════════════════════════════════════════════════════

def get_train_transform(img_size=IMAGE_SIZE):
    """
    Training pipeline:
        grayscale numpy array
        → PIL Image
        → resize to img_size×img_size
        → random small rotation (±5°)
        → random small shift (±5% of image width)
        → random slight zoom (90%–110%)
        → small brightness/contrast jitter
        → convert to tensor (float, 0.0–1.0)
        → normalise to (-1.0, +1.0)
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(
            degrees=5,                          # rotate ±5°
            translate=(0.05, 0.05),             # shift up to 5%
            scale=(0.93, 1.07),                 # zoom 93%–107%
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),                              # 20% chance of slight blur
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),                  # → float tensor, 0.0–1.0
        transforms.Normalize(mean=[0.5], std=[0.5]),  # → range -1.0 to +1.0
    ])


def get_eval_transform(img_size=IMAGE_SIZE):
    """
    Evaluation/test pipeline:
        No augmentation — just resize and normalise.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ═══════════════════════════════════════════════════════════════════
# DATASET CLASS  —  how to load one image at a time
# ═══════════════════════════════════════════════════════════════════
#
# PyTorch's DataLoader calls __getitem__(index) over and over.
# We just need to tell it: given an index, return (image_tensor, label_int).
#
# ═══════════════════════════════════════════════════════════════════

class NewaDataset(Dataset):
    """
    Custom Dataset for Newa script character images.

    Reads from one split directory, e.g. dataset_final/train/
    Each subdirectory name is the class label.
    """

    def __init__(self, split_dir, class_map, transform=None):
        """
        Args:
            split_dir : path to dataset_final/train  (or val / test)
            class_map : {"𑐎": 0, "𑐏": 1, ...}
            transform : image processing pipeline
        """
        self.split_dir = Path(split_dir)
        self.class_map = class_map
        self.transform = transform
        self.samples   = []   # list of (image_path, integer_label)

        for class_name, label in class_map.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((str(img_path), label))

        if not self.samples:
            raise RuntimeError(
                f"No images found under {self.split_dir}\n"
                f"Check your dataset_final/ structure."
            )

        n_classes = len(set(label for _, label in self.samples))
        print(f"  {self.split_dir.name:6s}: {len(self.samples):,} images  "
              f"| {n_classes} classes")

    def __len__(self):
        """How many images total in this split."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return one (image_tensor, label) pair.
        Called automatically by DataLoader.
        """
        img_path, label = self.samples[idx]

        # Read image as grayscale numpy array
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Cannot read image: {img_path}")

        # Convention: white background (255), dark ink (0)
        # If image is mostly dark, invert it
        if img.mean() < 127:
            img = 255 - img

        # Apply the transform pipeline
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_sample_weights(self):
        """
        Returns per-sample weights for WeightedRandomSampler.

        Why? Some classes may have more images than others.
        Weighted sampling makes sure rare classes appear as often
        as common ones during training — prevents the model from
        ignoring rare characters.
        """
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1

        total      = len(self.samples)
        n_classes  = len(self.class_map)

        # Weight = inverse frequency: rare class → higher weight → seen more often
        class_weight = {
            label: total / (n_classes * count)
            for label, count in label_counts.items()
        }

        return [class_weight[label] for _, label in self.samples]


# ═══════════════════════════════════════════════════════════════════
# DATALOADER FACTORY  —  the main function you call from train.py
# ═══════════════════════════════════════════════════════════════════

def get_dataloaders(dataset_dir=DATASET_DIR, img_size=IMAGE_SIZE,
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    weighted_sampler=True):
    """
    Build and return (train_loader, val_loader, test_loader, class_map).

    weighted_sampler=True  → oversample rare classes during training.
    Set False if all your classes already have equal image counts.
    """

    # ── Build or load class map ────────────────────────────────────
    map_path = Path(dataset_dir) / "class_map.json"
    if map_path.exists():
        class_map = load_class_map(map_path)
        print(f"Loaded class map: {len(class_map)} classes from {map_path}")
    else:
        class_map = build_class_map(dataset_dir, save_path=map_path)

    print(f"\nLoading datasets from {dataset_dir}/...")

    # ── Build Dataset objects ──────────────────────────────────────
    train_ds = NewaDataset(
        Path(dataset_dir) / "train",
        class_map,
        transform=get_train_transform(img_size),
    )
    val_ds = NewaDataset(
        Path(dataset_dir) / "val",
        class_map,
        transform=get_eval_transform(img_size),
    )
    test_ds = NewaDataset(
        Path(dataset_dir) / "test",
        class_map,
        transform=get_eval_transform(img_size),
    )

    # ── Build DataLoaders ──────────────────────────────────────────
    #
    # DataLoader wraps a Dataset and handles:
    #   - Batching (grouping images together)
    #   - Shuffling (random order each epoch)
    #   - Parallel loading (num_workers background threads)
    #   - GPU memory pinning (pin_memory=True speeds up GPU transfer)
    #
    if weighted_sampler:
        # Use weighted sampling to balance class frequencies
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,            # sampler replaces shuffle=True
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,             # drop last incomplete batch
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,                  # never shuffle evaluation data
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_map


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST  —  run this file directly to check everything works
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing dataset loading...\n")
    train_loader, val_loader, test_loader, class_map = get_dataloaders()

    # Grab one batch from each split
    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        imgs, labels = next(iter(loader))
        print(f"\n{name} batch:")
        print(f"  Image tensor shape : {imgs.shape}")
        #       should be [64, 1, 64, 64]
        #                  ↑   ↑  ↑   ↑
        #                batch ch  H   W
        print(f"  Value range        : {imgs.min():.2f} to {imgs.max():.2f}")
        #       should be roughly -1.0 to +1.0
        print(f"  Label range        : {labels.min().item()} to {labels.max().item()}")

    idx_to_class = {v: k for k, v in class_map.items()}
    print(f"\nFirst 5 classes: {[idx_to_class[i] for i in range(min(5, len(class_map)))]}")
    print("\ndataset.py OK — everything loads correctly.")