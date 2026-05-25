"""
recognize.py  —  Newa Character Recognition
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
Takes the character crop images produced by segment.py and runs
your trained OCR model on each one to predict which Newa character
it is.

HOW IT WORKS
────────────
1. Load the trained model from your checkpoint (.pth file)
2. For each character image:
   a. Resize to 64×64 (what the model was trained on)
   b. Normalise pixel values from [0,255] → [-1, +1]
      (same normalisation used during training)
   c. Feed through the neural network
   d. The model outputs 82 scores (one per character class)
   e. The highest score = predicted character
3. Map the class index back to the actual Newa Unicode character
   using the class_map saved inside the checkpoint

CONFIDENCE SCORE
────────────────
After the model outputs 82 raw scores ("logits"), we apply
softmax to turn them into probabilities that sum to 100%.
If the model says class 7 with 95% confidence, we trust it.
If it says 45% for class 7 and 40% for class 12, the image
is ambiguous — we flag it for review.

BATCH PROCESSING
────────────────
Instead of running images one at a time, we group them into
batches of 32. This is much faster because the GPU can process
32 images in parallel in the same time as one.

Run with:
    python transliteration/recognize.py \
        --segments output_segments/ \
        --checkpoint checkpoints/best_model.pth

Or run on a single image:
    python transliteration/recognize.py \
        --image path/to/single_char.png \
        --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent directory so we can import from ocr_model/
sys.path.insert(0, str(Path(__file__).parent.parent / "ocr_model"))
from model import build_model


# ═══════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═══════════════════════════════════════════════════════════════════

def preprocess_image(img_path: str, img_size: int = 64) -> torch.Tensor:
    """
    Load a character crop and prepare it for the model.

    Steps:
    1. Load as grayscale
    2. Resize to img_size × img_size
    3. Normalise to [0, 1] then standardise with mean=0.5, std=0.5
       → values in range [-1, +1]
       (must match the normalisation used during training)
    4. Add batch and channel dimensions: (H,W) → (1,1,H,W)
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Convert to float, normalise
    tensor = torch.from_numpy(img).float() / 255.0          # [0, 1]
    tensor = (tensor - 0.5) / 0.5                            # [-1, +1]
    tensor = tensor.unsqueeze(0).unsqueeze(0)               # (1,1,H,W)
    return tensor


# ═══════════════════════════════════════════════════════════════════
# DATASET FOR BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════

class CharacterCropDataset(Dataset):
    """
    A simple PyTorch dataset that reads character crop images from disk.
    Allows DataLoader to batch them efficiently.
    """

    def __init__(self, image_paths: list[str], img_size: int = 64):
        self.paths    = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Return blank image if file is corrupt/missing
            img = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255

        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0)   # add channel dim → (1, H, W)


# ═══════════════════════════════════════════════════════════════════
# LOAD MODEL FROM CHECKPOINT
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device):
    """
    Load the trained OCR model from a .pth checkpoint file.

    The checkpoint contains:
    - model_state:  the trained weights
    - class_map:    dict mapping index → character name/unicode
    - arch:         which architecture was used (convnet / resnet18)
    - img_size:     what image size the model expects
    - num_classes:  how many character classes (82)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    arch        = ckpt.get("arch",        "convnet")
    num_classes = ckpt.get("num_classes", 82)
    img_size    = ckpt.get("img_size",    64)
    class_map   = ckpt.get("class_map",   {})

    print(f"  Model: {arch}  |  {num_classes} classes  |  {img_size}px")
    print(f"  Best val accuracy: {ckpt.get('best_val_top1', '?')}%")

    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()   # Important: switches off dropout/batchnorm training behaviour

    # class_map may be stored as {index: char} or {char: index}
    # We need index → character name
    if class_map and isinstance(list(class_map.keys())[0], str):
        # Stored as {char_name: index} → invert it
        index_to_char = {v: k for k, v in class_map.items()}
    else:
        index_to_char = {int(k): v for k, v in class_map.items()}

    return model, index_to_char, img_size


# ═══════════════════════════════════════════════════════════════════
# RUN INFERENCE
# ═══════════════════════════════════════════════════════════════════

def recognize_batch(
    image_paths: list[str],
    model: torch.nn.Module,
    index_to_char: dict,
    img_size: int,
    device: torch.device,
    batch_size: int = 32,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """
    Run the OCR model on a list of character images.

    Returns a list of dicts, one per image:
    {
        "file":       "line_00_char_001.png",
        "predicted":  "KA",           ← character class name
        "unicode":    "𑑁",            ← actual Newa character
        "confidence": 0.97,           ← how sure the model is (0-1)
        "low_conf":   False,          ← True if confidence < threshold
        "top5":       [("KA",0.97), ("KHA",0.02), ...]
    }
    """
    dataset = CharacterCropDataset(image_paths, img_size)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    results = []
    img_idx = 0

    print(f"  Running OCR on {len(image_paths)} characters...")

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass: model outputs raw scores (logits)
            logits = model(batch)                          # (B, num_classes)

            # Softmax converts logits to probabilities (sum=1)
            probs  = F.softmax(logits, dim=1)              # (B, num_classes)

            # Get top-5 predictions for each image in the batch
            top5_probs, top5_indices = probs.topk(5, dim=1)

            for i in range(batch.size(0)):
                path = image_paths[img_idx]
                top5 = [
                    (index_to_char.get(top5_indices[i][j].item(), "?"),
                     round(top5_probs[i][j].item(), 4))
                    for j in range(5)
                ]

                best_char  = top5[0][0]
                best_conf  = top5[0][1]

                # Try to get the actual Unicode character
                # class name might be like "KA", "KHA", or directly "𑑁"
                # We'll resolve to Unicode in newa_to_devanagari.py
                results.append({
                    "file":       Path(path).name,
                    "predicted":  best_char,
                    "confidence": best_conf,
                    "low_conf":   best_conf < confidence_threshold,
                    "top5":       top5,
                })
                img_idx += 1

    # Summary
    low_conf_count = sum(1 for r in results if r["low_conf"])
    avg_conf       = sum(r["confidence"] for r in results) / len(results) if results else 0
    print(f"  Average confidence: {avg_conf:.1%}")
    print(f"  Low-confidence characters: {low_conf_count}/{len(results)}")

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN: RECOGNIZE ALL SEGMENTS
# ═══════════════════════════════════════════════════════════════════

def recognize_segments(
    segments_dir: str,
    checkpoint_path: str,
    output_json: str = None,
    batch_size: int  = 32,
    confidence_threshold: float = 0.3,
) -> list[dict]:
    """
    Recognize all character crops in a segments directory.
    Updates the segments_meta.json with predictions.
    """
    print(f"\n{'─'*55}")
    print(f"  Recognizing characters in: {segments_dir}")
    print(f"{'─'*55}")

    # Pick device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    # Load model
    print(f"  Loading model from: {checkpoint_path}")
    model, index_to_char, img_size = load_model(checkpoint_path, device)

    # Find all character images
    seg_path = Path(segments_dir)
    image_paths = sorted(seg_path.glob("line_*_char_*.png"))

    if not image_paths:
        print(f"  ERROR: No character images found in {segments_dir}")
        print(f"  Did you run segment.py first?")
        return []

    image_paths = [str(p) for p in image_paths]

    # Run recognition
    results = recognize_batch(
        image_paths, model, index_to_char, img_size,
        device, batch_size, confidence_threshold
    )

    # Load existing metadata and merge predictions
    meta_path = seg_path / "segments_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Build lookup from filename → prediction
        pred_lookup = {r["file"]: r for r in results}

        for char_meta in meta["characters"]:
            fname = char_meta["file"]
            if fname in pred_lookup:
                pred = pred_lookup[fname]
                char_meta["predicted"]  = pred["predicted"]
                char_meta["confidence"] = pred["confidence"]
                char_meta["low_conf"]   = pred["low_conf"]
                char_meta["top5"]       = pred["top5"]

        # Save updated metadata
        out_path = output_json or str(meta_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Updated metadata saved → {out_path}")

        return meta["characters"]

    else:
        # No metadata file — just save raw results
        out_path = output_json or str(seg_path / "predictions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Predictions saved → {out_path}")
        return results


def recognize_single(image_path: str, checkpoint_path: str):
    """Quick recognition of a single character image (for testing)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, index_to_char, img_size = load_model(checkpoint_path, device)
    results = recognize_batch([image_path], model, index_to_char,
                               img_size, device)
    r = results[0]
    print(f"\n  Image:      {image_path}")
    print(f"  Predicted:  {r['predicted']}  (confidence: {r['confidence']:.1%})")
    print(f"  Top 5:")
    for char, conf in r["top5"]:
        bar = "█" * int(conf * 20)
        print(f"    {char:12s}  {conf:.1%}  {bar}")
    return r


def parse_args():
    p = argparse.ArgumentParser(description="Recognize Newa characters with trained OCR model")
    p.add_argument("--segments",   help="Directory with character crops from segment.py")
    p.add_argument("--image",      help="Single character image (for testing)")
    p.add_argument("--checkpoint", required=True, help="Path to best_model.pth")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output",     help="Output JSON path (optional)")
    p.add_argument("--confidence", type=float, default=0.3,
                   help="Flag characters below this confidence (default: 0.3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.image:
        recognize_single(args.image, args.checkpoint)
    elif args.segments:
        recognize_segments(
            args.segments, args.checkpoint,
            args.output, args.batch_size, args.confidence
        )
    else:
        print("ERROR: provide --image or --segments")