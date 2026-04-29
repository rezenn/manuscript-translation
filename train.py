"""
train.py  —  Newa Script OCR
Trains the ResNet-18 CNN to classify 67 Newa characters.

Run this with:
    python train.py

What it does:
  1. Loads your dataset
  2. Builds the ResNet-18 model
  3. Trains for 30 epochs (you can change this)
  4. Saves the best model as "model_best.pth"
  5. Saves a training log as "training_log.csv"
  6. Shows a loss/accuracy graph at the end

Expected training time on an NVIDIA GPU: ~30-60 minutes for 30 epochs
"""

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm                   # progress bar
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import build_model

# ── Hyperparameters (settings you can tune) ───────────────────────────────────

NUM_EPOCHS    = 30          # how many times to go through the full training set
LEARNING_RATE = 1e-3        # how fast the model learns (start here, adjust later)
WEIGHT_DECAY  = 1e-4        # L2 regularization, prevents overfitting
SAVE_PATH     = "model_best.pth"    # where to save the best model
LOG_PATH      = "training_log.csv"  # where to save training history

# ── Device setup ──────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Helper functions ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Runs one full pass over the training data. Returns avg loss and accuracy."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()           # clear gradients from previous step
        outputs = model(images)         # forward pass: get predictions
        loss = criterion(outputs, labels)   # calculate how wrong we are
        loss.backward()                 # backward pass: calculate gradients
        optimizer.step()               # update model weights

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluates model on validation or test set. No gradient calculation."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():               # disable gradients for evaluation
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_history(history):
    """Saves a graph showing training and validation loss/accuracy over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train loss", color="blue")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train accuracy", color="blue")
    ax2.plot(epochs, history["val_acc"],   label="Val accuracy",   color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("\nGraph saved as training_curves.png")
    plt.show()


# ── Main training loop ────────────────────────────────────────────────────────

def main():
    # 1. Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    # Save class names so we can use them later in prediction
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Saved {len(class_names)} class names to class_names.txt")

    # 2. Build model
    print("\nBuilding model...")
    model = build_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # 3. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()           # standard for classification
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler: reduces LR by 10x if val accuracy stops improving
    # This helps squeeze out the last few % of accuracy
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",         # we want to maximize validation accuracy
        patience=5,         # wait 5 epochs before reducing LR
        factor=0.1,
        verbose=True
    )

    # 4. Training loop
    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # CSV log file
    log_file = open(LOG_PATH, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Update learning rate scheduler
        scheduler.step(val_acc)

        # Track history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Log to CSV
        log_writer.writerow([epoch, f"{train_loss:.4f}", f"{train_acc:.2f}",
                              f"{val_loss:.4f}", f"{val_acc:.2f}"])
        log_file.flush()

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": class_names,
            }, SAVE_PATH)
            saved_marker = "  ← BEST SAVED"
        else:
            saved_marker = ""

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"Train loss: {train_loss:.4f}  acc: {train_acc:.1f}% | "
            f"Val loss: {val_loss:.4f}  acc: {val_acc:.1f}%  "
            f"[{elapsed:.0f}s]{saved_marker}"
        )

    log_file.close()

    # 5. Final test evaluation using best saved model
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print("\nRunning final test evaluation with best model...")

    checkpoint = torch.load(SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.2f}%")

    # 6. Plot results
    plot_history(history)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"  Best model saved : {SAVE_PATH}")
    print(f"  Training log     : {LOG_PATH}")
    print(f"  Loss/acc graph   : training_curves.png")
    print(f"  Test accuracy    : {test_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()