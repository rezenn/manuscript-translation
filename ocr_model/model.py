"""
model.py  —  Newa Script OCR
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
Defines the neural network (CNN) that will look at a character image
and output a probability for each of your 82 Newa character classes.

WHAT IS A CNN?
──────────────
A Convolutional Neural Network (CNN) learns to recognize visual patterns.
It works in layers:

    Input image (64×64 grayscale)
         ↓
    Layer 1: finds simple edges (horizontal, vertical, diagonal)
         ↓
    Layer 2: combines edges into curves and corners
         ↓
    Layer 3: combines curves into parts (a stroke, a loop)
         ↓
    Layer 4+: recognizes whole character shapes
         ↓
    Final layer: outputs 82 numbers (one score per class)
         ↓
    Softmax: converts scores to probabilities (all sum to 100%)

WHICH MODEL DO WE USE?
──────────────────────
We offer two options:

1. NewaConvNet (our custom lightweight CNN)
   - Built from scratch, ~500k parameters
   - Trains fast even on a laptop CPU
   - Recommended for Phase 1 (synthetic data only)
   - Reaches ~80-90% accuracy on clean synthetic images

2. ResNet-18 (industry-standard pretrained network)
   - ~11 million parameters, pretrained on ImageNet
   - Better at handling the visual variety in real manuscripts
   - Recommended for Phase 3 (after adding handwritten + manuscript data)
   - Can reach 90-95%+ accuracy with diverse data

WHAT IS "PRETRAINED"?
──────────────────────
Training a CNN from scratch requires millions of images and hours/days.
ResNet-18 was already trained on 1.2 million ImageNet photos, so it
already knows how to detect edges, textures, and shapes.
We "fine-tune" it: replace only the final classification layer,
then train on our Newa data. Much faster and more accurate.
"""

import torch
import torch.nn as nn
from torchvision import models


# ═══════════════════════════════════════════════════════════════════
# OPTION 1: Custom Lightweight CNN  (start here — Phase 1)
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """
    A single "ConvBlock": Conv → BatchNorm → ReLU → (optional MaxPool)

    WHAT EACH PART DOES:
    - Conv2d:      scans the image with a small filter (e.g. 3×3)
                   to detect local patterns (edges, curves, etc.)
    - BatchNorm:   normalises activations → training is more stable
    - ReLU:        activation function. Replaces negatives with 0.
                   Adds "non-linearity" so the model can learn complex shapes.
    - MaxPool2d:   shrinks the feature map by 2× (takes the max in each 2×2 block)
                   Reduces computation and makes detection position-independent.
    """
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))   # halves spatial size
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class NewaConvNet(nn.Module):
    """
    Custom lightweight CNN for Newa character recognition.

    Architecture (input 1×64×64):
        ConvBlock(1→32,   pool) → feature map: 32×32×32
        ConvBlock(32→64,  pool) → feature map: 64×16×16
        ConvBlock(64→128, pool) → feature map: 128×8×8
        ConvBlock(128→256,pool) → feature map: 256×4×4
        GlobalAvgPool          → 256×1×1 → flatten to 256
        Dropout → Linear(256→256) → ReLU
        Dropout → Linear(256→num_classes)
    """

    def __init__(self, num_classes=82, dropout=0.4):
        super().__init__()

        # Feature extraction: turns image into abstract representations
        self.features = nn.Sequential(
            ConvBlock(1,   32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
            ConvBlock(128, 256, pool=True),
        )

        # Global Average Pooling: reduces each 4×4 feature map to a single number
        # Much better than flattening (avoids overfitting to position)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head: turns abstract features into class scores
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),          # randomly zeroes 40% of neurons
                                          # during training → prevents overfitting
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),  # final output: one score per class
        )

    def forward(self, x):
        """
        Forward pass: image tensor → class scores.
        Input shape:  [batch, 1, 64, 64]
        Output shape: [batch, num_classes]
        """
        x = self.features(x)         # [batch, 256, 4, 4]
        x = self.global_pool(x)      # [batch, 256, 1, 1]
        x = x.flatten(start_dim=1)   # [batch, 256]
        x = self.classifier(x)       # [batch, num_classes]
        return x


# ═══════════════════════════════════════════════════════════════════
# OPTION 2: ResNet-18  (upgrade to this in Phase 3)
# ═══════════════════════════════════════════════════════════════════

class NewaResNet(nn.Module):
    """
    ResNet-18 adapted for single-channel Newa character images.

    Changes from the standard ResNet-18:
    1. First conv layer: 3-channel RGB → 1-channel grayscale
       (average the pretrained RGB weights so we keep useful knowledge)
    2. Final FC layer: 1000 ImageNet classes → your num_classes
    """

    def __init__(self, num_classes=82, pretrained=True, dropout=0.5):
        super().__init__()

        # Load standard ResNet-18 with optional ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base    = models.resnet18(weights=weights)

        # ── Adapt first layer: RGB → grayscale ────────────────────
        orig_conv = base.conv1
        base.conv1 = nn.Conv2d(
            in_channels=1,                        # grayscale
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        if pretrained:
            # Average the 3 RGB weight channels into 1 grayscale channel
            # This preserves the learned edge detectors from ImageNet training
            with torch.no_grad():
                base.conv1.weight.copy_(
                    orig_conv.weight.mean(dim=1, keepdim=True)
                )

        # ── Replace classification head ────────────────────────────
        # Original: Linear(512 → 1000)
        # Ours:     Dropout + Linear(512 → num_classes)
        in_features = base.fc.in_features          # 512 for ResNet-18
        base.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        self.model = base

    def forward(self, x):
        return self.model(x)


# ═══════════════════════════════════════════════════════════════════
# FACTORY FUNCTION  —  what train.py actually calls
# ═══════════════════════════════════════════════════════════════════

def build_model(arch="convnet", num_classes=82, **kwargs):
    """
    Build and return the model.

    Args:
        arch        : "convnet" (fast, Phase 1) or "resnet18" (Phase 3)
        num_classes : how many character classes (default 82)
        **kwargs    : extra args passed to the model class
                      e.g. dropout=0.3

    Returns:
        PyTorch model (not yet moved to GPU — train.py does that)
    """
    if arch == "convnet":
        model = NewaConvNet(num_classes=num_classes, **kwargs)
    elif arch == "resnet18":
        model = NewaResNet(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose 'convnet' or 'resnet18'.")

    # Count and show parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {arch}  |  trainable params: {n_params:,}")
    return model


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Testing model builds...\n")

    for arch in ("convnet", "resnet18"):
        model = build_model(arch=arch, num_classes=82)
        dummy = torch.randn(4, 1, 64, 64)   # fake batch of 4 images
        out   = model(dummy)
        print(f"  {arch:10s}  input {tuple(dummy.shape)}  "
              f"→  output {tuple(out.shape)}")
        # output should be [4, 82] — 4 images, 82 class scores each

    print("\nmodel.py OK")