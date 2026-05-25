"""
train.py  —  Newa Script OCR
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
This is the main training script. It:
  1. Loads your image dataset
  2. Builds the neural network
  3. Trains it for N epochs (one epoch = seeing every training image once)
  4. After each epoch: measures accuracy on the validation set
  5. Saves the best model whenever validation accuracy improves
  6. Stops early if accuracy stops improving (saves time)
  7. Saves a training log and plots training curves

HOW TRAINING ACTUALLY WORKS (SIMPLIFIED)
──────────────────────────────────────────
Imagine the model starts with random weights — it has no idea what
Newa characters look like. Training adjusts those weights step by step:

  For each batch of 64 images:
    1. FORWARD PASS:  feed images through the model → get predictions
    2. LOSS:          measure how wrong the predictions are (CrossEntropy loss)
    3. BACKWARD PASS: calculate how to change each weight to reduce the loss
                      (this is "backpropagation" — calculus chain rule)
    4. UPDATE:        adjust weights by a tiny amount (learning rate × gradient)

  Repeat for every batch. After seeing every batch once → 1 epoch done.

LEARNING RATE
─────────────
Learning rate (lr) controls how big each weight adjustment is.
  - Too large: model overshoots, loss bounces around and never settles
  - Too small: training is very slow, may get stuck
  - We use: warmup for 3 epochs (slowly increase lr), then cosine decay
            (gradually decrease lr toward end of training)
  Good starting value: 3e-4 = 0.0003

WHAT TO EXPECT
──────────────
Phase 1 (synthetic data only, ~50k images):
  Epoch 1:  train acc ~30%, val acc ~25%  (model is still random)
  Epoch 5:  train acc ~70%, val acc ~65%
  Epoch 20: train acc ~90%, val acc ~82%
  Epoch 50: train acc ~95%, val acc ~85-90%  ← convergence

After adding handwritten data (Phase 2):
  Fine-tune from checkpoint, expect +5-10% val accuracy

Run with:
    python ocr_model/train.py

Or with custom settings:
    python ocr_model/train.py --arch convnet --epochs 50 --lr 3e-4
    python ocr_model/train.py --arch resnet18 --epochs 60 --resume checkpoints/best_model.pth
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — works without a display
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Import from our own files in ocr_model/
from dataset import get_dataloaders
from model import build_model


# ═══════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════

def accuracy_topk(outputs, labels, k=5):
    """
    Calculate top-k accuracy for a batch.

    Top-1: model's first guess is correct         (main metric)
    Top-5: correct answer is in model's top-5     (useful for debugging)
    """
    with torch.no_grad():
        batch_size = labels.size(0)
        k = min(k, outputs.size(1))               # handle k > num_classes
        _, pred = outputs.topk(k, dim=1)           # get top-k predictions
        pred    = pred.t()                         # transpose: [k, batch]
        correct = pred.eq(labels.view(1, -1))      # compare with true labels
        top1    = correct[:1].reshape(-1).float().sum() / batch_size * 100
        top5    = correct[:k].reshape(-1).float().sum() / batch_size * 100
    return top1.item(), top5.item()


def save_checkpoint(state, path):
    """Save a model checkpoint to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def plot_history(history, out_path="figures/training_curves.png"):
    """Save a loss + accuracy plot from the training history."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Newa OCR — Training History", fontsize=14, fontweight="bold")

    # Loss subplot
    axes[0].plot(epochs, [h["train_loss"] for h in history],
                 label="Train", color="#2196F3")
    axes[0].plot(epochs, [h["val_loss"] for h in history],
                 label="Val",   color="#F44336")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy subplot
    val_top1 = [h["val_top1"] for h in history]
    best_ep   = epochs[val_top1.index(max(val_top1))]
    axes[1].plot(epochs, [h["train_top1"] for h in history],
                 label="Train", color="#2196F3")
    axes[1].plot(epochs, val_top1,
                 label="Val",   color="#F44336")
    axes[1].axvline(best_ep, color="#4CAF50", linestyle="--", alpha=0.7,
                    label=f"Best ({max(val_top1):.1f}% @ ep {best_ep})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Top-1 Accuracy (%)")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# ONE EPOCH OF TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full training epoch.

    Returns: (average_loss, top1_accuracy, top5_accuracy)
    """
    model.train()   # enables dropout and batch norm in training mode

    total_loss = 0.0
    top1_sum   = 0.0
    top5_sum   = 0.0
    n_samples  = 0

    pbar = tqdm(loader, desc="  train", leave=False, ncols=80)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # ── Forward pass ──────────────────────────────────────────
        optimizer.zero_grad()           # clear gradients from last step
        outputs = model(images)         # model predicts class scores
        loss    = criterion(outputs, labels)   # measure error

        # ── Backward pass + weight update ─────────────────────────
        loss.backward()                 # compute gradients (backprop)
        # Clip gradients: prevents very large updates that destabilise training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()                # update weights

        # ── Track metrics ─────────────────────────────────────────
        bs = images.size(0)
        t1, t5 = accuracy_topk(outputs, labels)
        total_loss += loss.item() * bs
        top1_sum   += t1 * bs
        top5_sum   += t5 * bs
        n_samples  += bs

        pbar.set_postfix(
            loss=f"{total_loss/n_samples:.4f}",
            top1=f"{top1_sum/n_samples:.1f}%"
        )

    return total_loss / n_samples, top1_sum / n_samples, top5_sum / n_samples


# ═══════════════════════════════════════════════════════════════════
# ONE EPOCH OF EVALUATION (validation or test)
# ═══════════════════════════════════════════════════════════════════

def eval_epoch(model, loader, criterion, device):
    """
    Run evaluation (no gradient computation, no weight updates).
    Used for both validation (each epoch) and test (final).
    """
    model.eval()    # disables dropout, uses running stats for batch norm

    total_loss = 0.0
    top1_sum   = 0.0
    top5_sum   = 0.0
    n_samples  = 0

    with torch.no_grad():   # saves memory — no gradient graph needed
        pbar = tqdm(loader, desc="    val", leave=False, ncols=80)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            bs = images.size(0)
            t1, t5 = accuracy_topk(outputs, labels)
            total_loss += loss.item() * bs
            top1_sum   += t1 * bs
            top5_sum   += t5 * bs
            n_samples  += bs

    return total_loss / n_samples, top1_sum / n_samples, top5_sum / n_samples


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train(args):

    # ── Pick the best available device ────────────────────────────
    # CUDA  = NVIDIA GPU (fastest)
    # MPS   = Apple Silicon GPU (Mac M1/M2)
    # CPU   = fallback (slow but works)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU  (consider using a GPU for faster training)")

    # ── Load data ──────────────────────────────────────────────────
    print(f"\nLoading data from {args.data}/...")
    train_loader, val_loader, test_loader, class_map = get_dataloaders(
        dataset_dir=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    num_classes = len(class_map)

    # ── Build model ────────────────────────────────────────────────
    print(f"\nBuilding model: {args.arch}")
    model = build_model(arch=args.arch, num_classes=num_classes).to(device)

    # ── Resume from checkpoint (for Phase 2/3 fine-tuning) ────────
    start_epoch    = 1
    best_val_top1  = 0.0
    history        = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_val_top1  = ckpt.get("best_val_top1", 0.0)
        start_epoch    = ckpt.get("epoch", 0) + 1
        history        = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch-1} "
              f"(best val top-1 = {best_val_top1:.1f}%)")

    # ── Loss function ──────────────────────────────────────────────
    # CrossEntropyLoss: standard for multi-class classification
    # label_smoothing=0.05: instead of 100% sure of one class,
    #   the model targets 95% for correct + 0.07% for others.
    #   Helps with visually similar Newa characters.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ── Optimiser ─────────────────────────────────────────────────
    # AdamW: adaptive learning rates per parameter + weight decay
    # Better than plain SGD for this kind of task
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Learning rate schedule ─────────────────────────────────────
    # Phase 1 (warmup): lr grows from 10% to 100% over 3 epochs
    #   → avoids large gradient updates with random weights at start
    # Phase 2 (cosine): lr gradually decreases to lr/100 by the end
    #   → fine-tunes carefully in later epochs
    warmup_epochs = 0 if args.resume else 3
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                        total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_epochs, 1),
                                    eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer,
                                schedulers=[warmup, cosine],    
                                milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs,
                                    eta_min=args.lr * 0.01)

    # ── Setup output paths ─────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = ckpt_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_top1",
                         "val_loss",   "val_top1",   "lr"])

    # ── Training loop ──────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Training: {args.arch}  |  {args.epochs} epochs  "
          f"|  batch {args.batch_size}  |  lr {args.lr}")
    print(f"  Data: {args.data}/   |  {num_classes} classes")
    print(f"{'═'*65}\n")

    patience_count = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        # ── Train for one epoch ────────────────────────────────────
        tr_loss, tr_t1, tr_t5 = train_epoch(
            model, train_loader, criterion, optimizer, device)

        # ── Evaluate on validation set ─────────────────────────────
        va_loss, va_t1, va_t5 = eval_epoch(
            model, val_loader, criterion, device)

        # ── Step the learning rate schedule ───────────────────────
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # ── Print epoch summary ────────────────────────────────────
        print(f"Epoch {epoch:3d}  ({elapsed:.0f}s)  lr={current_lr:.2e}")
        print(f"  train  loss={tr_loss:.4f}  top1={tr_t1:.1f}%  top5={tr_t5:.1f}%")
        print(f"  val    loss={va_loss:.4f}  top1={va_t1:.1f}%  top5={va_t5:.1f}%")

        # ── Record history ─────────────────────────────────────────
        history.append(dict(
            epoch=epoch,
            train_loss=round(tr_loss, 4), train_top1=round(tr_t1, 2),
            val_loss=round(va_loss, 4),   val_top1=round(va_t1, 2),
        ))

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_t1:.2f}",
                             f"{va_loss:.4f}", f"{va_t1:.2f}",
                             f"{current_lr:.2e}"])

        # ── Save best model ────────────────────────────────────────
        if va_t1 > best_val_top1:
            best_val_top1  = va_t1
            patience_count = 0
            save_checkpoint({
                "epoch":         epoch,
                "arch":          args.arch,
                "num_classes":   num_classes,
                "img_size":      args.img_size,
                "model_state":   model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_top1": best_val_top1,
                "class_map":     class_map,
                "history":       history,
            }, ckpt_dir / "best_model.pth")
            print(f"  ✓ New best  val top-1={best_val_top1:.1f}%  → saved")
        else:
            patience_count += 1
            print(f"  No improvement  (patience {patience_count}/{args.patience})")

        # ── Periodic checkpoint (every N epochs) ──────────────────
        if epoch % args.save_every == 0:
            save_checkpoint({
                "epoch":       epoch,
                "arch":        args.arch,
                "num_classes": num_classes,
                "img_size":    args.img_size,
                "model_state": model.state_dict(),
                "class_map":   class_map,
                "history":     history,
            }, ckpt_dir / f"epoch_{epoch:04d}.pth")

        # ── Save history JSON ──────────────────────────────────────
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Early stopping ─────────────────────────────────────────
        if patience_count >= args.patience:
            print(f"\nEarly stopping: no improvement for "
                  f"{args.patience} epochs.")
            break

        print()

    # ── Final test evaluation ──────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Running final evaluation on TEST set (best model)...")
    best_ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    te_loss, te_t1, te_t5 = eval_epoch(model, test_loader, criterion, device)

    # ── Save training curves plot ──────────────────────────────────
    plot_history(history, out_path="figures/training_curves.png")

    print(f"\n{'═'*65}")
    print(f"  DONE")
    print(f"  Best val top-1  : {best_val_top1:.2f}%")
    print(f"  Test top-1      : {te_t1:.2f}%")
    print(f"  Test top-5      : {te_t5:.2f}%")
    print(f"  Checkpoint      : {ckpt_dir}/best_model.pth")
    print(f"  Training log    : {csv_path}")
    print(f"  Curves plot     : figures/training_curves.png")
    print(f"{'═'*65}\n")


# ═══════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Train the Newa OCR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PHASE 1 — Synthetic data only (start here):
  python ocr_model/train.py --arch convnet --epochs 50

PHASE 2 — After collecting handwritten samples:
  python ocr_model/train.py --arch convnet --resume checkpoints/best_model.pth --lr 5e-5 --epochs 30

PHASE 3 — After adding manuscript crops:
  python ocr_model/train.py --arch resnet18 --epochs 60
"""
    )
    p.add_argument("--data",           default="dataset_final",
                   help="Path to dataset_final/ (default: dataset_final)")
    p.add_argument("--arch",           default="convnet",
                   choices=["convnet", "resnet18"],
                   help="Model architecture (default: convnet)")
    p.add_argument("--epochs",         type=int,   default=50,
                   help="Number of training epochs (default: 50)")
    p.add_argument("--batch-size",     type=int,   default=64,
                   help="Batch size (default: 64, reduce to 32 if OOM)")
    p.add_argument("--lr",             type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--weight-decay",   type=float, default=1e-4,
                   help="Weight decay / L2 regularisation (default: 1e-4)")
    p.add_argument("--img-size",       type=int,   default=64,
                   help="Image size in pixels (default: 64)")
    p.add_argument("--workers",        type=int,   default=2,
                   help="DataLoader worker threads (default: 2; set 0 on Windows)")
    p.add_argument("--patience",       type=int,   default=12,
                   help="Early stopping patience in epochs (default: 12)")
    p.add_argument("--save-every",     type=int,   default=10,
                   help="Save a periodic checkpoint every N epochs (default: 10)")
    p.add_argument("--resume",         default=None,
                   help="Resume training from a .pth checkpoint")
    p.add_argument("--checkpoint-dir", default="checkpoints",
                   help="Where to save checkpoints (default: checkpoints/)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())