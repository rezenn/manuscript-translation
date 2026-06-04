"""
recognize.py  —  Newa Character Recognition (v2)
══════════════════════════════════════════════════════════════════

CHANGES vs v1
─────────────
• Fixed class_map inversion — v1 assumed keys were always ints which
  broke when the checkpoint stored them as strings.
• Passes line/char_idx through from segments_meta so the output JSON
  is always correctly ordered (line 0 → line N, left to right).
• Graceful skip of missing crop files (instead of crashing).
• Added --confidence flag default lowered to 0.25 — your model's
  avg confidence on real manuscripts is ~38%, so 0.30 was flagging
  too many real characters.

Run:
    python transliteration/recognize.py \\
        --segments output_segments/ \\
        --checkpoint checkpoints/best_model.pth --debug

    # Single image test:
    python transliteration/recognize.py \\
        --image output_segments/line_00_char_000.png \\
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

sys.path.insert(0, str(Path(__file__).parent.parent / "ocr_model"))
from model import build_model


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class CharacterCropDataset(Dataset):
    def __init__(self, image_paths: list, img_size: int = 64):
        self.paths    = image_paths
        self.img_size = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Blank white image as placeholder
            img = np.full((self.img_size, self.img_size), 255, dtype=np.uint8)

        img    = cv2.resize(img, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = (tensor - 0.5) / 0.5
        return tensor.unsqueeze(0)   # (1, H, W)


# ══════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device):
    """
    Load trained model from checkpoint.
    Handles both {int: name} and {name: int} class_map formats.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    arch        = ckpt.get("arch",        "convnet")
    num_classes = ckpt.get("num_classes", 67)
    img_size    = ckpt.get("img_size",    64)
    class_map   = ckpt.get("class_map",   {})

    print(f"  Model arch:    {arch}")
    print(f"  Num classes:   {num_classes}")
    print(f"  Image size:    {img_size}px")
    print(f"  Best val acc:  {ckpt.get('best_val_top1', '?')}%")

    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Normalise class_map to {int_index → class_name}
    if not class_map:
        print("  WARNING: checkpoint has no class_map — predictions will be indices")
        index_to_char = {}
    else:
        first_key = next(iter(class_map))
        if isinstance(first_key, str) and not first_key.isdigit():
            # Format: {"ka": 0, "kha": 1, ...} → invert
            index_to_char = {int(v): k for k, v in class_map.items()}
        else:
            # Format: {"0": "ka", "1": "kha", ...} or {0: "ka", ...}
            index_to_char = {int(k): v for k, v in class_map.items()}

    return model, index_to_char, img_size


# ══════════════════════════════════════════════════════════════════
# BATCH INFERENCE
# ══════════════════════════════════════════════════════════════════

def recognize_batch(
    image_paths: list,
    model: torch.nn.Module,
    index_to_char: dict,
    img_size: int,
    device: torch.device,
    batch_size: int = 32,
    confidence_threshold: float = 0.25,
) -> list:
    """
    Run model on all images. Returns list of prediction dicts.
    """
    dataset = CharacterCropDataset(image_paths, img_size)
    loader  = DataLoader(dataset, batch_size=batch_size,
                         shuffle=False, num_workers=0)

    results = []
    img_idx = 0

    print(f"  Running OCR on {len(image_paths)} characters...")

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            logits = model(batch)
            probs  = F.softmax(logits, dim=1)

            k = min(5, probs.shape[1])
            top_probs, top_idx = probs.topk(k, dim=1)

            for i in range(batch.size(0)):
                path  = image_paths[img_idx]
                top5  = [
                    (index_to_char.get(top_idx[i][j].item(), f"cls_{top_idx[i][j].item()}"),
                     round(top_probs[i][j].item(), 4))
                    for j in range(k)
                ]
                best_char = top5[0][0]
                best_conf = top5[0][1]

                results.append({
                    "file":       Path(path).name,
                    "predicted":  best_char,
                    "confidence": best_conf,
                    "low_conf":   best_conf < confidence_threshold,
                    "top5":       top5,
                })
                img_idx += 1

    low_n = sum(1 for r in results if r["low_conf"])
    avg_c = sum(r["confidence"] for r in results) / len(results) if results else 0
    print(f"  Avg confidence:    {avg_c:.1%}")
    print(f"  Low-conf chars:    {low_n}/{len(results)}")

    return results


# ══════════════════════════════════════════════════════════════════
# RECOGNIZE ALL SEGMENTS
# ══════════════════════════════════════════════════════════════════

def recognize_segments(
    segments_dir: str,
    checkpoint_path: str,
    output_json: str         = None,
    batch_size: int          = 32,
    confidence_threshold: float = 0.25,
) -> list:
    """
    Recognize all crops in segments_dir.
    Reads segments_meta.json, adds predictions, writes it back out.
    Returns the updated character list (ordered by line then char_idx).
    """
    print(f"\n{'─'*60}")
    print(f"  Recognizing: {segments_dir}")
    print(f"{'─'*60}")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    print(f"  Checkpoint: {checkpoint_path}")
    model, index_to_char, img_size = load_model(checkpoint_path, device)

    seg_path = Path(segments_dir)

    # Load metadata to get the correct ordering
    meta_path = seg_path / "segments_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        # Sort by line then char_idx to ensure correct text order
        char_list = sorted(meta["characters"],
                           key=lambda c: (c["line"], c["char_idx"]))
    else:
        # Fallback: discover files from disk
        found = sorted(seg_path.glob("line_*_char_*.png"))
        char_list = [{"file": p.name, "line": 0, "char_idx": i}
                     for i, p in enumerate(found)]
        meta = None

    if not char_list:
        print(f"  ERROR: No character crops in {segments_dir}")
        print(f"  Run segment.py first.")
        return []

    # Build ordered list of paths
    image_paths = []
    valid_chars = []
    for c in char_list:
        p = seg_path / c["file"]
        if p.exists():
            image_paths.append(str(p))
            valid_chars.append(c)
        else:
            print(f"  WARNING: missing crop {c['file']} — skipping")

    if not image_paths:
        print("  ERROR: all crop files missing")
        return []

    # Run recognition
    raw_results = recognize_batch(
        image_paths, model, index_to_char, img_size,
        device, batch_size, confidence_threshold
    )

    # Merge predictions back into char metadata
    for char_meta, pred in zip(valid_chars, raw_results):
        char_meta["predicted"]  = pred["predicted"]
        char_meta["confidence"] = pred["confidence"]
        char_meta["low_conf"]   = pred["low_conf"]
        char_meta["top5"]       = pred["top5"]

    # Save updated metadata
    out_path = output_json or str(meta_path)
    save_data = meta or {
        "source_image": str(segments_dir),
        "num_lines":    max((c["line"] for c in valid_chars), default=0) + 1,
        "num_chars":    len(valid_chars),
    }
    save_data["characters"] = valid_chars
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Updated metadata → {out_path}")

    return valid_chars


# ══════════════════════════════════════════════════════════════════
# SINGLE IMAGE (FOR TESTING)
# ══════════════════════════════════════════════════════════════════

def recognize_single(image_path: str, checkpoint_path: str):
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("cpu"))
    model, index_to_char, img_size = load_model(checkpoint_path, device)
    results = recognize_batch([image_path], model, index_to_char,
                               img_size, device)
    r = results[0]
    print(f"\n  Image:     {image_path}")
    print(f"  Predicted: {r['predicted']}  ({r['confidence']:.1%})")
    print(f"  Top 5:")
    for char, conf in r["top5"]:
        bar = "█" * int(conf * 30)
        print(f"    {char:20s}  {conf:.1%}  {bar}")
    return r


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Newa OCR recognition v2")
    p.add_argument("--segments",    help="Segments directory from segment.py")
    p.add_argument("--image",       help="Single crop image (for testing)")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--output",      help="Output JSON path (optional)")
    p.add_argument("--confidence",  type=float, default=0.25,
                   help="Flag chars below this confidence (default 0.25)")
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