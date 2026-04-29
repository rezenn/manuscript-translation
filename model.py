"""
model.py  —  Newa Script OCR
Defines the CNN model that will classify 67 Newa characters.

We use ResNet-18, a well-known architecture pretrained on ImageNet.
We modify it for:
  - 1-channel (grayscale) input  instead of 3-channel (RGB)
  - 67 output classes            instead of 1000 (ImageNet)

Why ResNet-18?
  - Fast to train (small enough for a single GPU)
  - Very accurate on small image classification tasks
  - Pretrained weights give us a huge head start
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 67

def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Returns a ResNet-18 model adapted for grayscale Newa script classification.

    pretrained=True  → starts with ImageNet weights (recommended, trains faster)
    pretrained=False → random weights (use only for experiments)
    """

    # Load ResNet-18 with pretrained ImageNet weights
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # ── Modify the first layer for grayscale (1 channel) ──────────────────────
    # Original ResNet expects 3-channel RGB input.
    # We replace the first conv layer to accept 1-channel grayscale.
    # We average the pretrained RGB weights across the 3 channels
    # so we don't lose the pretrained knowledge.
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,                          # grayscale
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )
    if pretrained:
        # Average the 3 RGB weight channels into 1
        model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

    # ── Modify the final layer for 67 classes ─────────────────────────────────
    # Original ResNet has 1000 output classes (ImageNet).
    # We replace the final fully-connected layer with 67 outputs.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model):
    """Prints total number of trainable parameters (just for info)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total:,}")
    return total


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    count_parameters(model)

    # Test with a fake batch: 4 grayscale 128x128 images
    dummy_input = torch.randn(4, 1, 128, 128)
    output = model(dummy_input)
    print(f"Input shape  : {dummy_input.shape}")    # [4, 1, 128, 128]
    print(f"Output shape : {output.shape}")         # [4, 67]
    print("\nmodel.py  OK — model builds and runs correctly.")