"""
fix_and_resume.py
─────────────────
Run this ONCE before resuming training to diagnose and fix the class map issue.

The problem:
  Your checkpoint was trained with a specific class_map (index → class name).
  If dataset.py rebuilds class_map.json on each run (alphabetical scan of folders),
  the order can change when new data is added — breaking the trained model.

The fix:
  1. Extract the class_map that was saved INSIDE the checkpoint
  2. Save it as dataset_final/class_map.json (overwriting any rebuilt version)
  3. Now training will use the EXACT same class ordering the model was trained on

Usage:
  python fix_and_resume.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import torch
from pathlib import Path


def fix(checkpoint_path: str):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Print what's inside the checkpoint
    print(f"\nCheckpoint contents:")
    print(f"  arch:          {ckpt.get('arch', '?')}")
    print(f"  num_classes:   {ckpt.get('num_classes', '?')}")
    print(f"  img_size:      {ckpt.get('img_size', '?')}")
    print(f"  best_val_top1: {ckpt.get('best_val_top1', '?')}%")
    print(f"  epoch:         {ckpt.get('epoch', '?')}")

    class_map = ckpt.get("class_map", {})
    if not class_map:
        print("\nERROR: No class_map found inside checkpoint!")
        print("This checkpoint was saved without class_map.")
        print("You will need to retrain from scratch OR manually reconstruct")
        print("the class_map from your original newa_classes.py in the same order.")
        return

    print(f"\nClass map inside checkpoint: {len(class_map)} classes")

    # Show the mapping
    # class_map may be {class_name: index} or {index: class_name}
    sample = list(class_map.items())[:5]
    print(f"  Sample entries: {sample}")

    # Determine direction
    first_key = list(class_map.keys())[0]
    if isinstance(first_key, str) and not first_key.isdigit():
        print("  Format: {class_name → index}")
        index_to_class = {str(v): k for k, v in class_map.items()}
    else:
        print("  Format: {index → class_name}")
        index_to_class = {str(k): v for k, v in class_map.items()}

    # Save the correct class_map to dataset_final/
    out_path = Path("dataset_final/class_map.json")
    out_path.parent.mkdir(exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Correct class_map saved → {out_path}")
    print(f"\nClass list (index → name):")
    for i in range(min(len(index_to_class), 20)):
        print(f"  {i:3d} → {index_to_class.get(str(i), '???')}")
    if len(index_to_class) > 20:
        print(f"  ... ({len(index_to_class)} total)")

    print(f"\n{'='*55}")
    print("NOW safe to resume training with:")
    print(f"  python ocr_model/train.py \\")
    print(f"    --arch {ckpt.get('arch', 'convnet')} \\")
    print(f"    --resume checkpoints/best_model.pth \\")
    print(f"    --lr 5e-5 \\")
    print(f"    --epochs 20 \\")
    print(f"    --patience 8")
    print(f"{'='*55}")

    # Also check if current dataset has different classes
    current_map_path = Path("dataset_final/class_map.json")
    print(f"\nVerification: re-reading saved file...")
    with open(current_map_path) as f:
        saved = json.load(f)
    print(f"  Saved class_map has {len(saved)} classes ✓")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    args = p.parse_args()
    fix(args.checkpoint)