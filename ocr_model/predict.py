"""
predict.py  —  Newa Script OCR
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
After training, use this to:

  1. TEST A SINGLE IMAGE:
       python ocr_model/predict.py --image path/to/char.png \
           --checkpoint checkpoints/best_model.pth

  2. EVALUATE THE FULL TEST SET (accuracy + per-class report):
       python ocr_model/predict.py \
           --checkpoint checkpoints/best_model.pth \
           --eval-test

  3. EVALUATE AND GENERATE ALL THESIS FIGURES:
       python ocr_model/predict.py \
           --checkpoint checkpoints/best_model.pth \
           --eval-test --figures

UNDERSTANDING THE OUTPUT
─────────────────────────
When you run on a single image, you get something like:

    Predictions for: dataset_final/test/𑐎/00001.png
    ─────────────────────────────────────────────────
     1.  𑐎    97.3%   ← TOP PREDICTION   ✓ correct
     2.  𑐏     1.8%
     3.  𑐐     0.5%
     4.  𑐑     0.2%
     5.  𑐒     0.2%

    Confidence 97.3% → model is very sure.
    If top confidence is below 40%, treat the prediction as uncertain.

WHAT IS CONFIDENCE?
──────────────────
After the model produces raw scores (logits), we apply "softmax"
which converts them to probabilities that sum to 100%.

    raw scores: [8.2,  2.1,  1.5, ...]
    softmax:    [97.3%, 1.8%, 0.5%, ...]
                 ↑ this is "confidence"

High confidence (>80%) → model is quite sure.
Low confidence (<40%)  → ambiguous character, may need review.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent dir to path so we can import from ocr_model/
sys.path.insert(0, str(Path(__file__).parent))
from dataset import NewaDataset, load_class_map, get_eval_transform
from model import build_model


# ═══════════════════════════════════════════════════════════════════
# LOAD A TRAINED CHECKPOINT
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    """
    Load a trained model from a .pth checkpoint file.

    The checkpoint contains:
      - model weights
      - class_map (which integer = which character)
      - architecture name
      - image size
      - training history

    Returns: (model, class_map, idx_to_class, img_size)
    """
    ckpt = torch.load(str(checkpoint_path), map_location=device)

    arch        = ckpt.get("arch", "convnet")
    num_classes = ckpt["num_classes"]
    img_size    = ckpt.get("img_size", 64)
    class_map   = ckpt["class_map"]              # {"𑐎": 0, "𑐏": 1, ...}
    idx_to_cls  = {v: k for k, v in class_map.items()}  # {0: "𑐎", 1: "𑐏", ...}

    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()   # IMPORTANT: evaluation mode disables dropout

    best_val = ckpt.get("best_val_top1", "N/A")
    print(f"Loaded {arch} | {num_classes} classes "
          f"| best val top-1: {best_val}%")

    return model, class_map, idx_to_cls, img_size


# ═══════════════════════════════════════════════════════════════════
# PREDICT A SINGLE IMAGE
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_image(image_path, model, idx_to_cls, img_size, device, top_k=5):
    """
    Run the model on one image and return top-k predictions.

    Returns:
        list of (class_name, confidence_percent) sorted by confidence
    """
    # Read image as grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read: {image_path}")

    # Ensure white background, dark ink (same convention as training data)
    if img.mean() < 127:
        img = 255 - img

    # Apply the same transform used during evaluation
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    tensor = tf(img).unsqueeze(0).to(device)   # [1, 1, 64, 64]

    # Forward pass
    logits = model(tensor)                     # [1, num_classes]
    probs  = F.softmax(logits, dim=1)[0]       # [num_classes]

    # Top-k results
    k = min(top_k, len(idx_to_cls))
    top_probs, top_idxs = probs.topk(k)

    return [
        (idx_to_cls[top_idxs[i].item()], top_probs[i].item() * 100)
        for i in range(k)
    ]


# ═══════════════════════════════════════════════════════════════════
# EVALUATE FULL TEST SET
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_set(model, test_loader, idx_to_cls, device, num_classes):
    """
    Run the model on the entire test set.

    Reports:
      - Overall top-1 and top-5 accuracy
      - Per-class accuracy (which characters are hardest)
      - Top confusion pairs (which characters get mixed up)
    """
    all_preds  = []
    all_labels = []

    print("Running test set evaluation...")
    for imgs, labels in tqdm(test_loader, desc="  test", ncols=70):
        imgs   = imgs.to(device)
        preds  = model(imgs).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Overall accuracy ───────────────────────────────────────────
    top1 = (all_preds == all_labels).mean() * 100
    correct = (all_preds == all_labels).sum()

    print(f"\n{'═'*55}")
    print(f"  TEST RESULTS")
    print(f"{'═'*55}")
    print(f"  Top-1 accuracy : {top1:.2f}%  "
          f"({correct}/{len(all_labels)} correct)")

    # ── Per-class accuracy ─────────────────────────────────────────
    per_class_acc = {}
    for idx in range(num_classes):
        mask = all_labels == idx
        if not mask.any():
            continue
        per_class_acc[idx] = (all_preds[mask] == idx).mean() * 100

    # Show the 15 hardest classes
    hardest = sorted(per_class_acc.items(), key=lambda x: x[1])[:15]
    print(f"\n  Hardest 15 classes:")
    print(f"  {'Character':25s}  {'Accuracy':>8s}")
    print(f"  {'─'*35}")
    for idx, acc in hardest:
        cls = idx_to_cls.get(idx, str(idx))
        bar = "█" * int(acc / 5)
        print(f"  {cls:25s}  {acc:6.1f}%  {bar}")

    # ── Top confusion pairs ────────────────────────────────────────
    # Which characters does the model most often mix up?
    confusion = {}
    for true, pred in zip(all_labels, all_preds):
        if true != pred:
            key = (int(true), int(pred))
            confusion[key] = confusion.get(key, 0) + 1

    top_conf = sorted(confusion.items(), key=lambda x: -x[1])[:10]
    print(f"\n  Top-10 confusions (true → predicted):")
    print(f"  {'True':25s}  {'Predicted':25s}  Count")
    print(f"  {'─'*60}")
    for (t, p), cnt in top_conf:
        tn = idx_to_cls.get(t, str(t))
        pn = idx_to_cls.get(p, str(p))
        print(f"  {tn:25s}  {pn:25s}  {cnt}×")

    print(f"{'─'*55}")

    return top1, per_class_acc


# ═══════════════════════════════════════════════════════════════════
# GENERATE THESIS FIGURES
# ═══════════════════════════════════════════════════════════════════

def generate_figures(model, test_loader, idx_to_cls,
                     device, num_classes, history, out_dir="figures"):
    """
    Generate publication-ready figures for your thesis:
      1. Training curves (loss + accuracy over epochs)
      2. Confusion matrix (which classes get confused)
      3. Per-class accuracy bar chart
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Training curves ──────────────────────────────────
    if history:
        epochs   = [h["epoch"]      for h in history]
        tr_loss  = [h["train_loss"] for h in history]
        va_loss  = [h["val_loss"]   for h in history]
        tr_top1  = [h["train_top1"] for h in history]
        va_top1  = [h["val_top1"]   for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Newa OCR — Training History",
                     fontsize=14, fontweight="bold")

        axes[0].plot(epochs, tr_loss, label="Train", color="#2196F3")
        axes[0].plot(epochs, va_loss, label="Val",   color="#F44336")
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        best_ep  = epochs[va_top1.index(max(va_top1))]
        axes[1].plot(epochs, tr_top1, label="Train", color="#2196F3")
        axes[1].plot(epochs, va_top1, label="Val",   color="#F44336")
        axes[1].axvline(best_ep, color="#4CAF50", linestyle="--", alpha=0.7,
                        label=f"Best ({max(va_top1):.1f}%)")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy")
        axes[1].legend(); axes[1].grid(alpha=0.3)

        plt.tight_layout()
        p = out_dir / "training_curves.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p}")

    # ── Figures 2 & 3: Confusion matrix + per-class accuracy ───────
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="  computing", ncols=70):
            preds = model(imgs.to(device)).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Figure 2: Confusion matrix (top 30 most confused classes)
    errors_per_class = {}
    for idx in range(num_classes):
        mask = all_labels == idx
        if mask.any():
            errors_per_class[idx] = (all_preds[mask] != idx).sum()

    top30 = sorted(errors_per_class, key=lambda x: -errors_per_class[x])[:30]
    top30_sorted = sorted(top30)
    n = len(top30_sorted)
    mat = np.zeros((n, n), dtype=int)
    idx_map = {cls: i for i, cls in enumerate(top30_sorted)}
    for t, p in zip(all_labels, all_preds):
        if t in idx_map and p in idx_map:
            mat[idx_map[t]][idx_map[p]] += 1

    labels_str = [idx_to_cls.get(i, str(i)) for i in top30_sorted]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(mat, xticklabels=labels_str, yticklabels=labels_str,
                cmap="Blues", ax=ax, linewidths=0.3,
                annot=(n <= 20), fmt="d")
    ax.set(xlabel="Predicted", ylabel="True",
           title=f"Confusion Matrix — Top {n} Most Confused Classes")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    p = out_dir / "confusion_matrix.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    # Figure 3: Per-class accuracy bar chart
    accs, names = [], []
    for idx in range(num_classes):
        mask = all_labels == idx
        if mask.any():
            accs.append((all_preds[mask] == idx).mean() * 100)
            names.append(idx_to_cls.get(idx, str(idx)))

    order  = np.argsort(accs)
    accs   = [accs[i]  for i in order]
    names  = [names[i] for i in order]
    colors = ["#F44336" if a < 70 else "#FF9800" if a < 90 else "#4CAF50"
              for a in accs]

    fig, ax = plt.subplots(figsize=(8, max(6, len(accs) * 0.28)))
    ax.barh(names, accs, color=colors, height=0.7)
    ax.axvline(np.mean(accs), color="black", linestyle="--", alpha=0.5,
               label=f"Mean {np.mean(accs):.1f}%")
    ax.set(xlabel="Top-1 Accuracy (%)",
           title="Per-Class Accuracy on Test Set")
    ax.set_xlim(0, 105)
    ax.legend(); ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    p = out_dir / "per_class_accuracy.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")
    print(f"\nAll figures saved to: {out_dir}/")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Newa OCR — predict single images or evaluate test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict one image
  python ocr_model/predict.py --image my_char.png \\
      --checkpoint checkpoints/best_model.pth

  # Full test set evaluation
  python ocr_model/predict.py --checkpoint checkpoints/best_model.pth \\
      --data dataset_final --eval-test

  # Generate thesis figures
  python ocr_model/predict.py --checkpoint checkpoints/best_model.pth \\
      --data dataset_final --eval-test --figures --out figures/
"""
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to best_model.pth")
    p.add_argument("--image",   default=None,
                   help="Path to a single image to predict")
    p.add_argument("--data",    default="dataset_final",
                   help="Dataset root (needed for --eval-test)")
    p.add_argument("--eval-test", action="store_true",
                   help="Evaluate on the full test set")
    p.add_argument("--figures",   action="store_true",
                   help="Generate training curves + confusion matrix figures")
    p.add_argument("--out",       default="figures",
                   help="Output directory for figures (default: figures/)")
    p.add_argument("--top-k",     type=int, default=5,
                   help="Show top-k predictions (default: 5)")
    p.add_argument("--batch-size",type=int, default=64)
    p.add_argument("--workers",   type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    # Load model
    model, class_map, idx_to_cls, img_size = load_model(
        args.checkpoint, device)
    num_classes = len(class_map)

    # ── Single image prediction ────────────────────────────────────
    if args.image:
        results = predict_image(
            args.image, model, idx_to_cls, img_size, device, top_k=args.top_k)

        print(f"\nPrediction for: {args.image}")
        print("─" * 50)
        for rank, (cls, conf) in enumerate(results, 1):
            marker = "  ← TOP PREDICTION" if rank == 1 else ""
            uncertain = "  [uncertain]" if rank == 1 and conf < 40 else ""
            print(f"  {rank}.  {cls:30s}  {conf:6.1f}%{marker}{uncertain}")

        top_cls, top_conf = results[0]
        note = "high confidence" if top_conf > 80 else \
               "moderate confidence" if top_conf > 50 else "uncertain"
        print(f"\nAnswer: {top_cls}  ({top_conf:.1f}% — {note})")

    # ── Full test set evaluation ───────────────────────────────────
    if args.eval_test or args.figures:
        tf  = get_eval_transform(img_size)
        ds  = NewaDataset(
            Path(args.data) / "test", class_map, transform=tf)
        loader = DataLoader(
            ds, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers)

        if args.eval_test:
            evaluate_test_set(model, loader, idx_to_cls, device, num_classes)

        if args.figures:
            # Load training history from checkpoint
            ckpt = torch.load(args.checkpoint, map_location=device)
            history = ckpt.get("history", [])
            if not history:
                # Try loading from history.json
                hist_path = Path(args.checkpoint).parent / "history.json"
                if hist_path.exists():
                    with open(hist_path) as f:
                        history = json.load(f)

            generate_figures(
                model, loader, idx_to_cls, device,
                num_classes, history, out_dir=args.out)

    # If neither --image nor --eval-test was given, show help
    if not args.image and not args.eval_test and not args.figures:
        print("Specify --image, --eval-test, or --figures. Run with --help for usage.")