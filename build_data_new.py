# step6_build_final_dataset.py
# ─────────────────────────────────────────────────────────────────
# Combines ALL sources and builds the final train/val/test split.
#
# Sources combined (with priority weights):
#   dataset_raw/augmented_noto/       weight 1.0
#   dataset_raw/augmented_ranjana/    weight 1.5  (fewer but important)
#   dataset_raw/handwritten_noto/     weight 2.0  (most valuable)
#   dataset_raw/handwritten_ranjana/  weight 2.5  (manuscript-style)
#   dataset_raw/manuscript_crops/     weight 3.0  (real manuscripts)
#
# Split: 80% train / 10% val / 10% test
#
# Output:  dataset_final/train/  val/  test/
#
# Usage:  python step6_build_final_dataset.py
# ─────────────────────────────────────────────────────────────────

import os, sys, shutil, random
from tqdm import tqdm
from collections import defaultdict
from utils import setup_utf8
setup_utf8()

sys.path.insert(0, os.path.dirname(__file__))
from newa_classes import NEWA_CHARACTERS

# ── Sources (add/remove as you collect data) ──────────────────────
SOURCES = [
    # (directory,                            weight)
    ("dataset_raw/augmented_noto",           1.0),
    ("dataset_raw/augmented_ranjana",        1.5),
    ("dataset_raw/augmented_prachalit1",     1.5),
    ("dataset_raw/handwritten_noto",         2.0),
    ("dataset_raw/handwritten_ranjana",      2.5),
    ("dataset_raw/manuscript_crops",         3.0),
]

OUTPUT  = "dataset_final"
SPLIT   = (0.80, 0.10, 0.10)
SEED    = 42


def collect_weighted_images(sources, class_name):
    """Collect all images for one class, applying weights by duplication."""
    all_images = []

    for src_dir, weight in sources:
        cls_dir = os.path.join(src_dir, class_name)
        if not os.path.exists(cls_dir):
            continue

        files = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.endswith('.png')
        ]

        if not files:
            continue

        # Apply weight by duplicating high-value sources
        int_w  = int(weight)
        frac_w = weight - int_w
        weighted = files * int_w
        if frac_w > 0:
            extra = random.sample(files,
                                  min(int(len(files)*frac_w)+1,
                                      len(files)))
            weighted += extra

        all_images.extend(weighted)

    return all_images


def build():
    random.seed(SEED)
    os.makedirs(OUTPUT, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT, split), exist_ok=True)

    # Discover all classes across all sources
    all_classes = set(NEWA_CHARACTERS.keys())
    # Also pick up any extra classes in source folders
    for src_dir, _ in SOURCES:
        if os.path.exists(src_dir):
            for c in os.listdir(src_dir):
                if os.path.isdir(os.path.join(src_dir, c)):
                    all_classes.add(c)

    print(f"Building final dataset for {len(all_classes)} classes...")

    stats = defaultdict(int)

    for class_name in tqdm(sorted(all_classes)):
        imgs = collect_weighted_images(SOURCES, class_name)
        if not imgs:
            continue

        random.shuffle(imgs)
        n = len(imgs)

        i1 = int(n * SPLIT[0])
        i2 = i1 + int(n * SPLIT[1])

        splits = {
            'train': imgs[:i1],
            'val':   imgs[i1:i2],
            'test':  imgs[i2:],
        }

        for split_name, files in splits.items():
            out_cls = os.path.join(OUTPUT, split_name, class_name)
            os.makedirs(out_cls, exist_ok=True)
            for k, src in enumerate(files):
                dst = os.path.join(out_cls, f"{k:05d}.png")
                shutil.copy(src, dst)
            stats[split_name] += len(files)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'='*60}")

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(OUTPUT, split)
        if not os.path.exists(split_dir):
            continue
        classes = [c for c in os.listdir(split_dir)
                   if os.path.isdir(os.path.join(split_dir, c))]
        total   = sum(
            len(os.listdir(os.path.join(split_dir, c)))
            for c in classes)
        avg     = total // len(classes) if classes else 0
        print(f"  {split:6s}: {total:7,} images  "
              f"({len(classes)} classes, avg {avg}/class)")

    total_all = sum(stats.values())
    print(f"  {'TOTAL':6s}: {total_all:7,} images")

    # Check for classes with too few images
    print(f"\n{'='*60}")
    print("CLASSES NEEDING MORE DATA (< 100 train images):")
    train_dir = os.path.join(OUTPUT, 'train')
    if os.path.exists(train_dir):
        low_classes = []
        for c in sorted(os.listdir(train_dir)):
            n = len(os.listdir(os.path.join(train_dir, c)))
            if n < 100:
                low_classes.append((c, n))
        if low_classes:
            for c, n in low_classes:
                print(f"  {c:20s}: {n} images")
        else:
            print("  All classes have ≥ 100 training images ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6 — BUILD FINAL DATASET")
    print("=" * 60)
    build()
    print(f"\nFinal dataset at: {OUTPUT}/")
    print("Ready to train your CNN!")