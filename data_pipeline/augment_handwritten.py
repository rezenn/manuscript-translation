"""
augment_handwritten.py  —  Newa Script OCR
============================================================
Augmentation pipeline for HANDWRITTEN character images.

These are real photos of handwritten Newa characters, so
augmentation must be CONSERVATIVE — we only apply transforms
that simulate natural real-world variation, not artistic effects.

WHY AUGMENT HANDWRITTEN DATA AT ALL?
  You have limited writers (maybe 3–10 people).
  Each person only filled 5 boxes per character.
  Without augmentation: ~5 samples per class per writer.
  With augmentation:   ~35 samples per class per writer.
  This prevents the model from memorising one person's handwriting.

WHAT WE DO (safe for handwritten):
  ✓ Small rotations        (writers tilt naturally)
  ✓ Small shear            (pen angle variation)
  ✓ Slight scale           (writing size variation)
  ✓ Brightness shift       (different lighting in photos)
  ✓ Contrast shift         (ink density variation)
  ✓ Thin noise             (paper texture, photo grain)
  ✓ Small translation      (character not always centred)

WHAT WE DON'T DO (would corrupt meaning):
  ✗ Horizontal flip        (many Newa chars are mirror pairs)
  ✗ Large rotations >10°   (character identity breaks)
  ✗ Elastic deformation    (distorts stroke shapes)
  ✗ Ink bleed / yellowing  (that's for manuscript style only)
  ✗ Heavy blur             (kills fine strokes)

USAGE:
  # Augment a single image and view the results:
  python augment_handwritten.py --demo path/to/image.png

  # Augment an entire dataset folder:
  python augment_handwritten.py --source dataset_raw/handwritten_noto
                                --output dataset_aug/handwritten_noto

  # Import and use in your own script:
  from augment_handwritten import augment_handwritten, augment_folder
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

IMG_SIZE = 128   # all images are this × this pixels

# ── Rotation ────────────────────────────────────────────────────
ROT_ANGLES   = [-8, -5, -3, 3, 5, 8]        # degrees

# ── Shear (slant) ───────────────────────────────────────────────
SHEAR_VALUES = [-0.10, -0.05, 0.05, 0.10]   # fraction of width

# ── Scale ───────────────────────────────────────────────────────
SCALE_VALUES = [0.88, 0.92, 1.08, 1.12]     # multiply IMG_SIZE

# ── Translation (shift) ─────────────────────────────────────────
TRANSLATE_PX = [-6, -3, 3, 6]               # pixels

# ── Brightness (multiply all pixels) ────────────────────────────
BRIGHTNESS_FACTORS = [0.80, 0.90, 1.10, 1.20]

# ── Contrast (alpha in: out = alpha*(in - 128) + 128) ───────────
CONTRAST_ALPHAS    = [0.80, 0.90, 1.10, 1.20]

# ── Noise (Gaussian std dev) ─────────────────────────────────────
NOISE_STDS = [5, 10]   # low noise only — just paper texture

# Which augmentations to apply (set False to skip any)
APPLY = {
    "rotation"   : True,
    "shear"      : True,
    "scale"      : True,
    "translate"  : True,
    "brightness" : True,
    "contrast"   : True,
    "noise"      : True,
}


# ═══════════════════════════════════════════════════════════════
# INDIVIDUAL TRANSFORM FUNCTIONS
# Each returns (suffix_string, augmented_image)
# suffix is used in the output filename so you know what was done
# ═══════════════════════════════════════════════════════════════

def apply_rotation(img, angle):
    """
    Rotate the character by `angle` degrees around the centre.
    Fills empty corners with white (background colour).
    Keeps the image at IMG_SIZE × IMG_SIZE.

    Real writers tilt characters by a few degrees naturally.
    This teaches the model to handle that variation.
    """
    h, w   = img.shape
    centre = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255)    # white background
    return f"rot{angle:+d}", rotated


def apply_shear(img, shear):
    """
    Apply horizontal shear — the character leans left or right.
    Simulates pen angle variation between writers.

    Shear matrix:  [1  shear]   applied to x-coordinate
                   [0    1  ]
    """
    h, w = img.shape
    # Shift so the character doesn't slide off-frame
    shift = abs(shear) * h / 2
    M = np.float32([
        [1, shear, -shift if shear > 0 else 0],
        [0, 1,      0]
    ])
    sheared = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255)
    suffix = f"sh{int(shear*100):+d}"
    return suffix, sheared


def apply_scale(img, scale_factor):
    """
    Scale the character up or down, then crop/pad back to IMG_SIZE.
    Simulates different writing sizes — some writers write big,
    some write small.
    """
    h, w       = img.shape
    new_size   = int(h * scale_factor)
    # Resize
    resized = cv2.resize(img, (new_size, new_size),
                         interpolation=cv2.INTER_AREA
                         if scale_factor < 1
                         else cv2.INTER_LINEAR)
    # Create white canvas of original size
    canvas = np.full((h, w), 255, dtype=np.uint8)
    # Centre-paste
    if scale_factor < 1:
        # smaller: paste in centre
        pad_y = (h - new_size) // 2
        pad_x = (w - new_size) // 2
        canvas[pad_y:pad_y+new_size, pad_x:pad_x+new_size] = resized
    else:
        # larger: crop centre
        start_y = (new_size - h) // 2
        start_x = (new_size - w) // 2
        canvas = resized[start_y:start_y+h, start_x:start_x+w]
        if canvas.shape != (h, w):
            canvas = cv2.resize(canvas, (w, h))
    suffix = f"sc{int(scale_factor*100)}"
    return suffix, canvas


def apply_translation(img, dx, dy=0):
    """
    Shift the character by (dx, dy) pixels.
    Fills the gap with white.

    Characters aren't always perfectly centred in the crop box.
    This teaches the model to handle off-centre characters.
    """
    h, w = img.shape
    M    = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255)
    suffix = f"tr{dx:+d}"
    return suffix, shifted


def apply_brightness(img, factor):
    """
    Multiply all pixel values by `factor`.
    factor < 1  → darker  (simulates dim lighting)
    factor > 1  → lighter (simulates bright/overexposed photo)

    Clamps to [0, 255] to avoid wrap-around artifacts.
    """
    bright = np.clip(img.astype(np.float32) * factor, 0, 255)\
               .astype(np.uint8)
    suffix = f"br{int(factor*100)}"
    return suffix, bright


def apply_contrast(img, alpha):
    """
    Adjust contrast around the midpoint (128).
    out = alpha * (in - 128) + 128

    alpha < 1  → lower contrast (ink looks faded)
    alpha > 1  → higher contrast (ink looks bolder)

    Simulates variation in pen ink density and paper whiteness.
    """
    adjusted = np.clip(
        alpha * (img.astype(np.float32) - 128) + 128,
        0, 255).astype(np.uint8)
    suffix = f"co{int(alpha*100)}"
    return suffix, adjusted


def apply_noise(img, std):
    """
    Add Gaussian noise to the image.
    Simulates:
      • Paper grain
      • JPEG compression artifacts in the photo
      • Subtle ink texture variation

    std controls noise intensity — we keep this small
    so strokes stay legible.
    """
    noise  = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy  = np.clip(img.astype(np.float32) + noise, 0, 255)\
               .astype(np.uint8)
    suffix = f"ns{std}"
    return suffix, noisy


# ═══════════════════════════════════════════════════════════════
# MAIN AUGMENTATION FUNCTION
# ═══════════════════════════════════════════════════════════════

def augment_handwritten(img, apply=None):
    """
    Apply all enabled augmentations to one grayscale crop image.

    Args:
        img   : grayscale numpy array, shape (IMG_SIZE, IMG_SIZE)
        apply : dict of {augmentation_name: bool} overrides.
                Defaults to the global APPLY config above.

    Returns:
        List of (suffix, augmented_image) tuples.
        Each tuple = one training sample.

    Example:
        results = augment_handwritten(my_crop)
        # results might contain 20–25 (suffix, image) pairs
        for suffix, aug_img in results:
            cv2.imwrite(f"ka_aug_{suffix}.png", aug_img)
    """
    if apply is None:
        apply = APPLY

    results = []

    if apply.get("rotation", True):
        for angle in ROT_ANGLES:
            results.append(apply_rotation(img, angle))

    if apply.get("shear", True):
        for shear in SHEAR_VALUES:
            results.append(apply_shear(img, shear))

    if apply.get("scale", True):
        for sf in SCALE_VALUES:
            results.append(apply_scale(img, sf))

    if apply.get("translate", True):
        for dx in TRANSLATE_PX:
            results.append(apply_translation(img, dx))

    if apply.get("brightness", True):
        for factor in BRIGHTNESS_FACTORS:
            results.append(apply_brightness(img, factor))

    if apply.get("contrast", True):
        for alpha in CONTRAST_ALPHAS:
            results.append(apply_contrast(img, alpha))

    if apply.get("noise", True):
        for std in NOISE_STDS:
            results.append(apply_noise(img, std))

    return results


# ═══════════════════════════════════════════════════════════════
# FOLDER AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def augment_folder(source_dir, output_dir, overwrite=False):
    """
    Augments an entire dataset folder.

    Expects source_dir to have the ImageFolder layout:
        source_dir/
            ka/   img1.png  img2.png ...
            kha/  img1.png  ...
            ...

    Saves augmented images alongside originals in output_dir
    using the same layout.

    Args:
        source_dir : path to raw handwritten crops
                     e.g. "dataset_raw/handwritten_noto"
        output_dir : where augmented data goes
                     e.g. "dataset_aug/handwritten_noto"
                     (can be the same as source_dir to augment in-place)
        overwrite  : if True, re-augment even if output files exist

    File naming:
        Original:   hw_noto_wrajju_0003.png
        Augmented:  hw_noto_wrajju_0003_rot+7.png
                    hw_noto_wrajju_0003_sh-10.png
                    etc.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f"✗ Source not found: {source_dir}")
        return

    class_dirs = sorted([d for d in source_dir.iterdir()
                         if d.is_dir()])
    if not class_dirs:
        print(f"✗ No class folders found in {source_dir}")
        return

    print(f"\nHandwritten augmentation")
    print(f"  Source : {source_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Classes: {len(class_dirs)}")

    total_orig = 0
    total_aug  = 0

    for cls_dir in class_dirs:
        class_name = cls_dir.name
        out_cls    = output_dir / class_name
        out_cls.mkdir(parents=True, exist_ok=True)

        png_files = sorted(cls_dir.glob("*.png"))
        if not png_files:
            continue

        for img_path in png_files:
            # Copy original to output if not already there
            orig_out = out_cls / img_path.name
            if not orig_out.exists() or overwrite:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(str(orig_out), img)
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            total_orig += 1
            stem = img_path.stem

            # Generate augmented versions
            aug_versions = augment_handwritten(img)
            for suffix, aug_img in aug_versions:
                aug_name = f"{stem}_{suffix}.png"
                aug_path = out_cls / aug_name
                if not aug_path.exists() or overwrite:
                    cv2.imwrite(str(aug_path), aug_img)
                    total_aug += 1

        print(f"  {class_name:<20}: "
              f"{len(png_files)} orig → "
              f"{len(png_files) * (1 + len(augment_handwritten(np.ones((IMG_SIZE,IMG_SIZE),np.uint8)*200)))} total")

    print(f"\n  Done.")
    print(f"  Original images : {total_orig}")
    print(f"  Augmented added : {total_aug}")
    print(f"  Total in output : {total_orig + total_aug}")


# ═══════════════════════════════════════════════════════════════
# DEMO — visualise all augmentations on one image
# ═══════════════════════════════════════════════════════════════

def demo(image_path):
    """
    Loads one character image and saves a grid showing every
    augmented version side by side.

    Output: augmentation_demo_handwritten.jpg
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    versions = [("original", img)] + augment_handwritten(img)
    print(f"Total versions including original: {len(versions)}")

    # Arrange in a grid
    cols    = 7
    rows    = (len(versions) + cols - 1) // cols
    cell_sz = IMG_SIZE + 24   # image + label space
    grid    = np.ones((rows * cell_sz, cols * cell_sz), np.uint8) * 230

    for idx, (suffix, v_img) in enumerate(versions):
        r   = idx // cols
        c   = idx %  cols
        y0  = r * cell_sz
        x0  = c * cell_sz

        # Paste image
        grid[y0:y0+IMG_SIZE, x0:x0+IMG_SIZE] = v_img

        # Draw label
        cv2.putText(grid, suffix,
                    (x0 + 2, y0 + IMG_SIZE + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, 0, 1)

    out_path = "augmentation_demo_handwritten.jpg"
    cv2.imwrite(out_path, grid)
    print(f"Demo saved: {out_path}")
    print("Open this file to see every augmentation applied.")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Handwritten Newa character augmentation")
    parser.add_argument("--demo",   metavar="IMAGE",
        help="Run demo on a single image")
    parser.add_argument("--source", metavar="DIR",
        help="Source folder of handwritten crops")
    parser.add_argument("--output", metavar="DIR",
        help="Output folder for augmented dataset")
    parser.add_argument("--overwrite", action="store_true",
        help="Re-generate even if files already exist")
    args = parser.parse_args()

    if args.demo:
        demo(args.demo)
    elif args.source and args.output:
        augment_folder(args.source, args.output, args.overwrite)
    else:
        print(__doc__)
        print("\nExamples:")
        print("  python augment_handwritten.py --demo ka/hw_noto_wrajju_0000.png")
        print("  python augment_handwritten.py \\")
        print("      --source dataset_raw/handwritten_noto \\")
        print("      --output dataset_aug/handwritten_noto")