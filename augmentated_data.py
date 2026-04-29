

import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
from utils import setup_utf8
setup_utf8()

sys.path.insert(0, os.path.dirname(__file__))

# ── Augmentation jobs ─────────────────────────────────────────────
# (source_dir, output_dir, pipeline_name, augments_per_image)
JOBS = [
    ("dataset_raw/synthetic_noto",
     "dataset_raw/augmented_noto",
     "noto", 60),
    ("dataset_raw/synthetic_ranjana",
     "dataset_raw/augmented_ranjana",
     "ranjana", 80),

    # ("dataset_raw/synthetic_ranjana",
    #  "dataset_raw/augmented_ranjana",
    #  "ranjana", 70),   
     
    # ("dataset_raw/synthetic_prachalit1",
    #  "dataset_raw/augmented_prachalit1",
    #  "ranjana", 70),   
]


# ─────────────────────────────────────────────────────────────────
# Augmentation functions (no albumentations needed)
# ─────────────────────────────────────────────────────────────────

def aug_rotate(img, limit=12):
    angle = np.random.uniform(-limit, limit)
    h, w  = img.shape
    M     = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderValue=255)


def aug_rotate_manuscript(img, limit=18):
    """Heavier rotation for manuscript style."""
    return aug_rotate(img, limit=limit)


def aug_shear(img, shear_range=0.15):
    """Simulates calligraphic slant."""
    h, w = img.shape
    shear = np.random.uniform(-shear_range, shear_range)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderValue=255)


def aug_blur(img, max_k=5):
    k = np.random.choice([3, 5])
    if k > max_k:
        k = max_k
    return cv2.GaussianBlur(img, (k, k), 0)


def aug_ink_bleed(img):
    """Simulates ink spreading into manuscript paper."""
    k = np.random.choice([3, 5])
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    # Dilate dark pixels (ink spreads)
    kernel   = np.ones((2, 2), np.uint8)
    inv      = cv2.bitwise_not(blurred)
    dilated  = cv2.dilate(inv, kernel, iterations=1)
    return cv2.bitwise_not(dilated)


def aug_noise(img, std=15):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise,
                   0, 255).astype(np.uint8)


def aug_noise_heavy(img, std=35):
    return aug_noise(img, std=std)


def aug_brightness(img, low=0.75, high=1.25):
    f   = np.random.uniform(low, high)
    out = np.clip(img.astype(np.float32) * f, 0, 255)
    return out.astype(np.uint8)


def aug_brightness_dark(img):
    """Simulates faded/dark manuscript ink."""
    return aug_brightness(img, low=0.55, high=0.95)


def aug_perspective(img, strength=0.06):
    h, w = img.shape
    m    = int(w * strength)
    src  = np.float32([[0,0],[w,0],[0,h],[w,h]])
    dst  = np.float32([
        [np.random.randint(0, m+1), np.random.randint(0, m+1)],
        [w-np.random.randint(0,m+1), np.random.randint(0,m+1)],
        [np.random.randint(0, m+1), h-np.random.randint(0,m+1)],
        [w-np.random.randint(0,m+1), h-np.random.randint(0,m+1)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255)


def aug_elastic(img, alpha=3, sigma=12):
    """Simulates hand stroke variation."""
    h, w = img.shape
    dx   = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32) * alpha,
        (0, 0), sigma)
    dy   = cv2.GaussianBlur(
        np.random.randn(h, w).astype(np.float32) * alpha,
        (0, 0), sigma)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    mx   = np.clip(x + dx, 0, w-1).astype(np.float32)
    my   = np.clip(y + dy, 0, h-1).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderValue=255)


def aug_scale(img, low=0.85, high=1.15):
    h, w     = img.shape
    factor   = np.random.uniform(low, high)
    new_h    = int(h * factor)
    new_w    = int(w * factor)
    resized  = cv2.resize(img, (new_w, new_h))
    canvas   = np.full((h, w), 255, np.uint8)
    y0 = max(0, (h - new_h) // 2)
    x0 = max(0, (w - new_w) // 2)
    y1 = min(h, y0 + new_h)
    x1 = min(w, x0 + new_w)
    ry1 = y1 - y0
    rx1 = x1 - x0
    canvas[y0:y1, x0:x1] = resized[:ry1, :rx1]
    return canvas


# ── Pipeline definitions ──────────────────────────────────────────

NOTO_PIPELINE = [
    aug_rotate,
    aug_blur,
    aug_noise,
    aug_brightness,
    aug_perspective,
    aug_scale,
]

RANJANA_PIPELINE = [
    aug_rotate_manuscript,
    aug_shear,
    aug_ink_bleed,
    aug_noise_heavy,
    aug_brightness_dark,
    aug_perspective,
    aug_elastic,
    aug_scale,
]

PIPELINES = {
    "noto":    NOTO_PIPELINE,
    "ranjana": RANJANA_PIPELINE,
}


def augment_one(img, pipeline):
    """Apply 2–4 random augmentations from pipeline."""
    n_aug  = np.random.randint(2, min(5, len(pipeline)+1))
    chosen = np.random.choice(len(pipeline),
                              size=n_aug, replace=False)
    result = img.copy()
    for idx in chosen:
        result = pipeline[idx](result)
    return result


def is_blank(img, threshold=250, min_dark=8):
    return (img < threshold).sum() < min_dark


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 3 — AUGMENTATION")
    print("=" * 60)

    for src_dir, out_dir, pipeline_name, n_aug in JOBS:
        print(f"\n[{pipeline_name}]  {src_dir} → {out_dir}")

        if not os.path.exists(src_dir):
            print(f"  ✗ Source not found — run step2 first")
            continue

        pipeline = PIPELINES[pipeline_name]
        os.makedirs(out_dir, exist_ok=True)

        classes = [c for c in os.listdir(src_dir)
                   if os.path.isdir(os.path.join(src_dir, c))]
        print(f"  Classes: {len(classes)}  "
              f"Augments per image: {n_aug}")

        total_saved = 0

        for class_name in tqdm(classes,
                               desc=f"Aug [{pipeline_name}]"):
            src_cls = os.path.join(src_dir, class_name)
            out_cls = os.path.join(out_dir, class_name)
            os.makedirs(out_cls, exist_ok=True)

            src_files = [f for f in os.listdir(src_cls)
                         if f.endswith('.png')]

            # Copy originals
            for fname in src_files:
                img = cv2.imread(
                    os.path.join(src_cls, fname),
                    cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    cv2.imwrite(
                        os.path.join(out_cls, f"orig_{fname}"),
                        img)

            # Generate augmented
            aug_count = 0
            target    = n_aug * len(src_files)

            while aug_count < target:
                for fname in src_files:
                    if aug_count >= target:
                        break
                    img = cv2.imread(
                        os.path.join(src_cls, fname),
                        cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    aug = augment_one(img, pipeline)

                    if not is_blank(aug):
                        cv2.imwrite(
                            os.path.join(out_cls,
                                f"aug_{aug_count:05d}.png"),
                            aug)
                        aug_count  += 1
                        total_saved += 1

        print(f"  Total saved: {total_saved:,}")
        per_cls = total_saved // len(classes) if classes else 0
        print(f"  Per class  : ~{per_cls}")

    print(f"\n{'='*60}")
    print("Augmentation complete.")
    print("Check dataset_raw/augmented_noto/")
    print("Check dataset_raw/augmented_ranjana/")