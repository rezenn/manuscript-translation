"""
debug_crops.py — Visual diagnosis of segmentation + recognition quality
══════════════════════════════════════════════════════════════════════

WHY THIS EXISTS
───────────────
Line-level OCR is failing (mostly ण / ग / ख garbage) even though the
trained model scores ~96% val accuracy and 90%+ confidence on clean
single-character test images.

That mismatch means: the crops segment.py is producing from real
manuscript LINES do not look like the clean single-character crops
the model was trained/tested on. This script makes that visible by
laying out every crop from a line next to its filename, predicted
class, and confidence — as an image grid you can actually inspect.

USE THIS BEFORE touching the model. If most boxes are slivers of a
stroke, half a character, or two characters glued together, the bug
is in segment.py's box-finding, not in the network.

Run:
    python transliteration/debug_crops.py --segments output_segments --checkpoint checkpoints/best_model.pth
    python transliteration/debug_crops.py --segments output_segments --checkpoint checkpoints/best_model.pth --line 3
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from recognize import load_model, recognize_batch


def build_montage(segments_dir, checkpoint_path, line_filter=None,
                   out_path="debug_crops_montage.png", cell=96):
    seg_path = Path(segments_dir)
    meta_path = seg_path / "segments_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No segments_meta.json in {segments_dir}. Run segment.py first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chars = sorted(meta["characters"], key=lambda c: (c["line"], c["char_idx"]))
    if line_filter is not None:
        chars = [c for c in chars if c["line"] == line_filter]
    if not chars:
        print("No crops match that line filter.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, index_to_char, img_size = load_model(checkpoint_path, device)

    paths = [str(seg_path / c["file"]) for c in chars]
    results = recognize_batch(paths, model, index_to_char, img_size, device,
                               confidence_threshold=0.5)

    # Group by line for row-wise layout
    by_line = {}
    for c, r in zip(chars, results):
        by_line.setdefault(c["line"], []).append((c, r))

    rows = []
    for line_idx in sorted(by_line):
        items = by_line[line_idx]
        row_cells = []
        for c, r in items:
            img = cv2.imread(str(seg_path / c["file"]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.full((cell, cell), 200, dtype=np.uint8)
            img = cv2.resize(img, (cell, cell - 22))
            canvas = np.full((cell, cell), 255, dtype=np.uint8)
            canvas[:cell - 22, :] = img
            label = f"{r['predicted']}:{r['confidence']:.2f}"
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            color = (0, 150, 0) if r["confidence"] >= 0.5 else (0, 0, 220)
            cv2.putText(canvas_bgr, label, (2, cell - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)
            cv2.rectangle(canvas_bgr, (0, 0), (cell - 1, cell - 1), (200, 200, 200), 1)
            row_cells.append(canvas_bgr)
        # pad row to common width if needed and stack horizontally
        row_img = np.hstack(row_cells)
        rows.append(row_img)

    max_w = max(r.shape[1] for r in rows)
    padded_rows = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)
    montage = np.vstack(padded_rows)
    cv2.imwrite(out_path, montage)

    print(f"\n  Montage saved → {out_path}")
    print(f"  {len(chars)} crops across {len(by_line)} line(s)")
    print("\n  WHAT TO LOOK FOR:")
    print("  - If a crop shows only PART of a character (a single stroke, a loose")
    print("    loop, a diacritic with no base) → segmentation is over-fragmenting.")
    print("  - If a crop shows TWO characters glued together → segmentation is")
    print("    under-fragmenting (boxes not split where they should be).")
    print("  - If most crops look like clean, complete single characters but the")
    print("    prediction is still wrong → that's an actual model/training problem.")
    print("  - Red labels = confidence below 0.5 (likely junk/fragment).")

    # Quick fragment heuristic on box geometry
    widths = [c["width"] for c in chars]
    heights = [c["height"] for c in chars]
    med_w, med_h = float(np.median(widths)), float(np.median(heights))
    tiny = [c for c in chars if c["width"] < med_w * 0.45 or c["height"] < med_h * 0.45]
    huge = [c for c in chars if c["width"] > med_w * 2.2]
    print(f"\n  Median box size: {med_w:.0f}x{med_h:.0f}px")
    print(f"  Suspiciously SMALL boxes (likely fragments): {len(tiny)}/{len(chars)} "
          f"({100*len(tiny)/len(chars):.0f}%)")
    print(f"  Suspiciously WIDE boxes (likely merged characters): {len(huge)}/{len(chars)} "
          f"({100*len(huge)/len(chars):.0f}%)")
    if len(tiny) / len(chars) > 0.25:
        print("  → High fragment rate. This strongly explains the garbage line output.")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize segmentation crops + predictions")
    p.add_argument("--segments", required=True, help="Folder with segments_meta.json + crop PNGs")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    p.add_argument("--line", type=int, default=None, help="Only show this line index")
    p.add_argument("--out", default="debug_crops_montage.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_montage(args.segments, args.checkpoint, args.line, args.out)