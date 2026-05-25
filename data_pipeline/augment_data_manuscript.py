"""
augment_manuscript.py  —  Newa Script OCR
============================================================
Augmentation pipeline for MANUSCRIPT character crops.

These are crops from historical Newa manuscript pages
(like the image you shared — Prachalit calligraphy on aged
parchment). The goal is to make the model robust to all the
degradation effects present in real manuscripts.

WHY MANUSCRIPT AUGMENTATION IS DIFFERENT:
  Manuscript images have:
  • Aged, yellowed parchment backgrounds (not white)
  • Ink that has bled, faded, or cracked over centuries
  • Uneven lighting and shadows from curved/folded pages
  • Wormholes, stains, and tears
  • Calligraphic strokes (thicker, more curved than modern writing)
  • Red decorative marks and borders (visible in your image)

  The model trained only on clean synthetic or modern handwritten
  data will FAIL on these. Manuscript augmentation bridges that gap.

WHAT WE DO (heavy, manuscript-specific):
  ✓ Parchment background    (yellow-brown aged paper texture)
  ✓ Ink fading              (characters look washed out)
  ✓ Ink bleeding            (strokes spread into paper)
  ✓ Elastic deformation     (calligraphic stroke variation)
  ✓ Uneven illumination     (shadow from curved page)
  ✓ Noise + grain           (photo of aged paper)
  ✓ Blur                    (out-of-focus manuscript scan)
  ✓ Red ink marks           (decorative elements in manuscripts)
  ✓ Stain patches           (age spots, water damage)
  ✓ Rotation + shear        (character orientation on page)

USAGE:
  # Demo on one crop:
  python augment_manuscript.py --demo path/to/crop.png

  # Augment entire manuscript crops folder:
  python augment_manuscript.py --source dataset_raw/manuscript_data
                               --output dataset_raw/manuscript_data

  # Import and use:
  from augment_manuscript import augment_manuscript, augment_folder
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from scipy.ndimage import map_coordinates, gaussian_filter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

IMG_SIZE = 128

# How many augmented copies to make per original crop
# Manuscript crops are rare and precious — make more copies
COPIES_PER_IMAGE = 20

# Random seed for reproducibility (set None for true random)
RANDOM_SEED = None

# Which effects to apply (set False to disable any)
APPLY = {
    "parchment_bg"       : True,
    "ink_fading"         : True,
    "ink_bleeding"       : True,
    "elastic_deform"     : True,
    "uneven_illumination": True,
    "noise"              : True,
    "blur"               : True,
    "stain"              : True,
    "rotation"           : True,
    "shear"              : True,
}


# ═══════════════════════════════════════════════════════════════
# EFFECT FUNCTIONS
# Each takes a COLOUR (BGR) image and returns a BGR image.
# Manuscript augmentation works in colour because the aged
# parchment background is yellowish-brown, not pure white.
# ═══════════════════════════════════════════════════════════════

def to_color(img_gray):
    """Convert grayscale to BGR colour for manuscript effects."""
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


def to_gray(img_bgr):
    """Convert back to grayscale after colour effects."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ── Effect 1: Parchment background ──────────────────────────────

def apply_parchment_background(img_bgr):
    """
    Replaces the plain white background with aged parchment colour.

    Real manuscripts have backgrounds ranging from cream/ivory to
    deep yellow-brown depending on the age and material.
    We generate a smoothly varying background (not flat) to
    simulate the natural texture variation of manuscript paper.

    The character (dark ink pixels) is preserved.
    """
    h, w = img_bgr.shape[:2]

    # Generate a random parchment colour in the yellow-brown range
    # BGR: blue low, green medium, red high → warm yellow-brown
    base_b = np.random.randint(140, 200)  # blue channel
    base_g = np.random.randint(160, 220)  # green channel
    base_r = np.random.randint(180, 240)  # red channel

    # Create a smooth random gradient for texture
    # (small noise, then blur a LOT to get smooth variation)
    noise  = np.random.randint(0, 30, (h, w), dtype=np.uint8).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=20)

    # Build the 3-channel parchment background
    bg = np.zeros((h, w, 3), dtype=np.float32)
    bg[:, :, 0] = np.clip(base_b + smooth - 15, 100, 255)  # B
    bg[:, :, 1] = np.clip(base_g + smooth - 10, 120, 255)  # G
    bg[:, :, 2] = np.clip(base_r + smooth,       140, 255) # R

    # Identify background pixels: where original image is near-white
    # (> 200 on all channels)
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_mask    = (gray > 200).astype(np.float32)

    # Blend: background gets parchment colour, ink stays dark
    result = np.zeros_like(img_bgr, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = (bg_mask       * bg[:, :, c] +
                           (1-bg_mask)   * img_bgr[:, :, c].astype(float))

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Effect 2: Ink fading ─────────────────────────────────────────

def apply_ink_fading(img_bgr, fade_strength=None):
    """
    Simulates ink that has faded over centuries.

    Faded ink appears lighter — the dark strokes become grey
    rather than black. Some parts of a stroke may fade more
    than others (patchy fading).

    fade_strength: 0.0 = no fade, 1.0 = completely invisible
    """
    if fade_strength is None:
        fade_strength = np.random.uniform(0.15, 0.55)

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Create a patchy fade map — some areas fade more
    fade_map = np.random.uniform(0, 1, (h // 8, w // 8)).astype(np.float32)
    fade_map = cv2.resize(fade_map, (w, h))
    fade_map = cv2.GaussianBlur(fade_map, (0, 0), sigmaX=5)
    fade_map = (fade_map * fade_strength).clip(0, 1)

    # Lift dark pixels towards the background grey
    result = img_bgr.astype(np.float32)
    for c in range(3):
        # Pixels get pushed towards a light gray (simulating faded ink)
        result[:, :, c] += fade_map * (180 - result[:, :, c])

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Effect 3: Ink bleeding ───────────────────────────────────────

def apply_ink_bleeding(img_bgr, amount=None):
    """
    Simulates ink that has spread (bled) into the paper fibres.

    Over time (and especially on thin paper), ink soaks outwards
    from the original stroke. This makes strokes appear thicker
    and slightly fuzzy at the edges.

    We model this with directional dilation + blending.
    """
    if amount is None:
        amount = np.random.uniform(0.3, 0.8)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Identify dark strokes
    _, ink_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Expand the ink (bleeding = dilation)
    k_size  = np.random.choice([3, 5])
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bled    = cv2.dilate(ink_mask, kernel, iterations=1)

    # Soften the edge of the bleed
    bled_soft = cv2.GaussianBlur(bled.astype(np.float32), (5, 5), 0) / 255.0

    # Blend bleed with original
    bleed_color = np.array([40, 30, 20], dtype=np.float32)  # dark ink
    result = img_bgr.astype(np.float32)
    for c in range(3):
        result[:, :, c] = (result[:, :, c] * (1 - amount * bled_soft) +
                           bleed_color[c]  * (amount * bled_soft))

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Effect 4: Elastic deformation ───────────────────────────────

def apply_elastic_deform(img_bgr, alpha=None, sigma=None):
    """
    Applies elastic (smooth random) deformation to the image.

    This simulates:
      • Natural variation in calligraphic brush strokes
      • Slight warping from a curved/folded manuscript page
      • Variation between different scribes' writing styles

    alpha : controls deformation strength (displacement magnitude)
    sigma : controls smoothness (larger = smoother, more natural)

    We use a random displacement field that is smoothed with
    Gaussian blur — this gives organic-looking deformations
    rather than harsh distortions.
    """
    if alpha is None:
        alpha = np.random.uniform(3, 8)
    if sigma is None:
        sigma = np.random.uniform(3, 6)

    h, w = img_bgr.shape[:2]

    # Random displacement fields for x and y
    dx = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1), sigma) * alpha
    dy = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1), sigma) * alpha

    # Build coordinate maps
    x, y     = np.meshgrid(np.arange(w), np.arange(h))
    new_x    = np.clip(x + dx, 0, w - 1)
    new_y    = np.clip(y + dy, 0, h - 1)

    result = np.zeros_like(img_bgr)
    for c in range(3):
        result[:, :, c] = map_coordinates(
            img_bgr[:, :, c],
            [new_y.ravel(), new_x.ravel()],
            order=1, mode='nearest'
        ).reshape(h, w)

    return result.astype(np.uint8)


# ── Effect 5: Uneven illumination ────────────────────────────────

def apply_uneven_illumination(img_bgr):
    """
    Simulates shadows and uneven lighting across the manuscript page.

    When photographing a manuscript:
      • The page may be curved → one side gets more light
      • The camera flash creates a bright spot in the centre
      • Natural light comes from one direction

    We model this with a large smooth gradient multiply.
    """
    h, w = img_bgr.shape[:2]

    # Random gradient direction
    gradient_type = np.random.choice(
        ["horizontal", "vertical", "radial", "corner"])

    mask = np.ones((h, w), dtype=np.float32)

    if gradient_type == "horizontal":
        # Fade left-to-right or right-to-left
        col    = np.linspace(0.7, 1.0, w)
        if np.random.rand() > 0.5:
            col = col[::-1]
        mask   = np.tile(col, (h, 1))

    elif gradient_type == "vertical":
        row    = np.linspace(0.7, 1.0, h)
        if np.random.rand() > 0.5:
            row = row[::-1]
        mask   = np.tile(row.reshape(-1, 1), (1, w))

    elif gradient_type == "radial":
        # Bright centre, dark edges (typical camera flash)
        cx, cy = w // 2, h // 2
        Y, X   = np.ogrid[:h, :w]
        dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask   = 1.0 - 0.3 * (dist / dist.max())

    elif gradient_type == "corner":
        # Dark in one corner
        cx = np.random.choice([0, w])
        cy = np.random.choice([0, h])
        Y, X   = np.ogrid[:h, :w]
        dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask   = 0.7 + 0.3 * (dist / dist.max())

    # Smooth the mask
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=30).clip(0.5, 1.2)

    result = img_bgr.astype(np.float32)
    for c in range(3):
        result[:, :, c] *= mask

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Effect 6: Paper grain / noise ────────────────────────────────

def apply_grain_noise(img_bgr, intensity=None):
    """
    Adds realistic paper grain texture.

    Manuscript paper has visible grain (fibre texture) and
    photos of manuscripts have additional camera sensor noise.

    We use a mix of:
      • Fine Gaussian noise (camera grain)
      • Coarser speckle noise (paper fibre variation)
    """
    if intensity is None:
        intensity = np.random.uniform(5, 20)

    h, w = img_bgr.shape[:2]

    # Fine Gaussian noise
    fine  = np.random.normal(0, intensity, (h, w, 3)).astype(np.float32)

    # Coarser speckle (low-res noise, upscaled)
    coarse_h = h // 4
    coarse_w = w // 4
    coarse   = np.random.normal(
        0, intensity * 0.5, (coarse_h, coarse_w, 3)).astype(np.float32)
    coarse   = cv2.resize(coarse, (w, h))

    noise    = fine + coarse
    result   = np.clip(img_bgr.astype(np.float32) + noise, 0, 255)
    return result.astype(np.uint8)


# ── Effect 7: Blur (scan focus) ──────────────────────────────────

def apply_blur(img_bgr, blur_type=None):
    """
    Applies mild blurring to simulate:
      • Slightly out-of-focus camera when photographing manuscript
      • Scanning at slightly lower DPI
      • Motion blur from hand-held photography

    We keep blur very mild — the character must remain legible.
    """
    if blur_type is None:
        blur_type = np.random.choice(
            ["gaussian", "motion", "none"], p=[0.5, 0.3, 0.2])

    if blur_type == "none":
        return img_bgr

    elif blur_type == "gaussian":
        ksize = np.random.choice([3, 5])
        return cv2.GaussianBlur(img_bgr, (ksize, ksize), 0)

    elif blur_type == "motion":
        # Horizontal motion blur — simulates camera movement
        k   = np.random.randint(2, 5)
        M   = np.zeros((k, k))
        M[k // 2, :] = 1.0 / k
        return cv2.filter2D(img_bgr, -1, M)

    return img_bgr


# ── Effect 8: Age stains ─────────────────────────────────────────

def apply_stain(img_bgr):
    """
    Adds random age spots and stain patches to the image.

    Manuscript pages often have:
      • Water damage stains (irregular brown patches)
      • Mold/foxing spots (small dark dots)
      • Wormholes (small dark circles)
      • Ink smears from adjacent pages

    We add 1–4 random semi-transparent patches.
    These should not completely obscure the character.
    """
    result = img_bgr.astype(np.float32)
    h, w   = img_bgr.shape[:2]

    n_stains = np.random.randint(1, 4)
    for _ in range(n_stains):
        # Random stain colour (brown, dark, orange tones)
        r = np.random.randint(60, 160)
        g = np.random.randint(40, 120)
        b = np.random.randint(20, 80)
        stain_color = np.array([b, g, r], dtype=np.float32)

        # Random position and size
        cx  = np.random.randint(10, w - 10)
        cy  = np.random.randint(10, h - 10)
        rx  = np.random.randint(5, 25)
        ry  = np.random.randint(5, 20)
        opacity = np.random.uniform(0.1, 0.35)

        # Create an elliptical stain mask
        stain_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(stain_mask,
                    (cx, cy), (rx, ry),
                    np.random.randint(0, 180),
                    0, 360, 1.0, -1)

        # Blur the edges of the stain for a natural look
        stain_mask = cv2.GaussianBlur(stain_mask, (15, 15), 0)
        stain_mask *= opacity

        # Blend stain colour into the image
        for c in range(3):
            result[:, :, c] = (result[:, :, c] * (1 - stain_mask) +
                               stain_color[c]  * stain_mask)

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Effect 9: Rotation (manuscript range) ────────────────────────

def apply_manuscript_rotation(img_bgr, angle=None):
    """
    Rotate the crop.

    Manuscript characters can be at varying angles because:
      • The scribe tilted their page
      • The manuscript page was photographed at an angle
      • Characters on curved pages appear tilted

    We allow slightly larger rotations than handwritten (±12°)
    and fill with a realistic parchment colour, not plain white.
    """
    if angle is None:
        angle = np.random.uniform(-12, 12)

    h, w    = img_bgr.shape[:2]
    centre  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(centre, angle, 1.0)

    # Fill with a parchment-like colour
    fill_b = np.random.randint(160, 210)
    fill_g = np.random.randint(170, 220)
    fill_r = np.random.randint(190, 235)
    fill   = (int(fill_b), int(fill_g), int(fill_r))

    rotated = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill)
    return rotated


# ── Effect 10: Shear (calligraphic slant) ────────────────────────

def apply_manuscript_shear(img_bgr, shear=None):
    """
    Horizontal shear — simulates the natural slant of
    calligraphic script written at different angles.

    Prachalit manuscript writing often has a consistent slant
    that varies between scribes and time periods.
    """
    if shear is None:
        shear = np.random.uniform(-0.15, 0.15)

    h, w  = img_bgr.shape[:2]
    shift = abs(shear) * h / 2
    M     = np.float32([
        [1, shear, -shift if shear > 0 else 0],
        [0, 1,      0]
    ])

    fill_b = np.random.randint(160, 210)
    fill_g = np.random.randint(170, 220)
    fill_r = np.random.randint(190, 235)
    fill   = (int(fill_b), int(fill_g), int(fill_r))

    sheared = cv2.warpAffine(
        img_bgr, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=fill)
    return sheared


# ═══════════════════════════════════════════════════════════════
# AUGMENTATION PIPELINE
# Combines multiple effects in a sensible order.
# ═══════════════════════════════════════════════════════════════

def augment_manuscript_once(img_gray, apply=None, seed=None):
    """
    Applies a randomised combination of manuscript effects
    to produce ONE augmented version of a character crop.

    Call this multiple times (COPIES_PER_IMAGE times) to get
    a diverse set of augmented versions.

    Args:
        img_gray : grayscale numpy array (IMG_SIZE × IMG_SIZE)
        apply    : dict of effect enable/disable overrides
        seed     : optional random seed for reproducibility

    Returns:
        Augmented grayscale image (IMG_SIZE × IMG_SIZE)

    ORDER OF EFFECTS:
      1. Geometric (rotation, shear, elastic) — applied first,
         before adding visual effects, so the geometry transform
         doesn't smear the visual effects.
      2. Background (parchment) — applied to the geometric result.
      3. Ink effects (fading, bleeding) — applied to the image
         with realistic background already in place.
      4. Lighting (uneven illumination) — applied after ink
         effects since lighting affects both ink and background.
      5. Noise and blur — applied last, simulating camera/scan
         quality on top of everything else.
      6. Stains — applied last so they sit "on top" of the page.
    """
    if seed is not None:
        np.random.seed(seed)
    if apply is None:
        apply = APPLY

    img = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))

    # ── 1. Geometric transforms ─────────────────────────────────
    img_bgr = to_color(img)

    if apply.get("elastic_deform") and np.random.rand() > 0.3:
        img_bgr = apply_elastic_deform(img_bgr)

    if apply.get("rotation") and np.random.rand() > 0.3:
        img_bgr = apply_manuscript_rotation(img_bgr)

    if apply.get("shear") and np.random.rand() > 0.4:
        img_bgr = apply_manuscript_shear(img_bgr)

    # ── 2. Parchment background ──────────────────────────────────
    if apply.get("parchment_bg"):
        img_bgr = apply_parchment_background(img_bgr)

    # ── 3. Ink effects ───────────────────────────────────────────
    if apply.get("ink_fading") and np.random.rand() > 0.3:
        img_bgr = apply_ink_fading(img_bgr)

    if apply.get("ink_bleeding") and np.random.rand() > 0.4:
        img_bgr = apply_ink_bleeding(img_bgr)

    # ── 4. Uneven illumination ───────────────────────────────────
    if apply.get("uneven_illumination") and np.random.rand() > 0.4:
        img_bgr = apply_uneven_illumination(img_bgr)

    # ── 5. Noise and blur ────────────────────────────────────────
    if apply.get("noise") and np.random.rand() > 0.3:
        img_bgr = apply_grain_noise(img_bgr)

    if apply.get("blur") and np.random.rand() > 0.5:
        img_bgr = apply_blur(img_bgr)

    # ── 6. Stains ────────────────────────────────────────────────
    if apply.get("stain") and np.random.rand() > 0.6:
        img_bgr = apply_stain(img_bgr)

    # Convert back to grayscale for model training
    result = to_gray(img_bgr)
    return result


def augment_manuscript(img_gray, n_copies=COPIES_PER_IMAGE, apply=None):
    """
    Generate multiple augmented versions of one manuscript crop.

    Args:
        img_gray : grayscale source image
        n_copies : how many augmented versions to generate
        apply    : optional effect enable/disable dict

    Returns:
        List of (suffix, augmented_image) tuples.
        suffix is "ms_aug_{i:02d}" for each copy.

    Example:
        versions = augment_manuscript(crop_img, n_copies=20)
        for suffix, aug in versions:
            cv2.imwrite(f"ka_{suffix}.png", aug)
    """
    results = []
    for i in range(n_copies):
        aug = augment_manuscript_once(img_gray, apply=apply)
        results.append((f"ms_aug_{i:02d}", aug))
    return results


# ═══════════════════════════════════════════════════════════════
# FOLDER AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def augment_folder(source_dir, output_dir,
                   n_copies=COPIES_PER_IMAGE, overwrite=False):
    """
    Augments an entire manuscript crops folder.

    Expects:
        source_dir/
            ka/    ms_0000.png  ms_0001.png ...
            kha/   ms_0000.png ...

    Saves:
        output_dir/
            ka/    ms_0000.png              ← original copy
                   ms_0000_ms_aug_00.png   ← augmented copies
                   ms_0000_ms_aug_01.png
                   ...

    Args:
        source_dir : path to labeled manuscript crops
        output_dir : where augmented data goes
        n_copies   : augmented versions per original
        overwrite  : regenerate even if files exist
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f"✗ Source not found: {source_dir}")
        return

    class_dirs = sorted([d for d in source_dir.iterdir()
                         if d.is_dir() and d.name != "unlabeled"])
    if not class_dirs:
        print(f"✗ No class folders found in {source_dir}")
        return

    print(f"\nManuscript augmentation")
    print(f"  Source  : {source_dir}")
    print(f"  Output  : {output_dir}")
    print(f"  Classes : {len(class_dirs)}")
    print(f"  Copies  : {n_copies} per original")

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
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Copy original
            orig_out = out_cls / img_path.name
            if not orig_out.exists() or overwrite:
                cv2.imwrite(str(orig_out), img)
            total_orig += 1

            # Generate augmented copies
            aug_versions = augment_manuscript(img, n_copies=n_copies)
            stem = img_path.stem
            for suffix, aug_img in aug_versions:
                aug_path = out_cls / f"{stem}_{suffix}.png"
                if not aug_path.exists() or overwrite:
                    cv2.imwrite(str(aug_path), aug_img)
                    total_aug += 1

        print(f"  {class_name:<20}: "
              f"{len(png_files)} orig → "
              f"{len(png_files) * (1 + n_copies)} total")

    print(f"\n  Done.")
    print(f"  Original images  : {total_orig}")
    print(f"  Augmented added  : {total_aug}")
    print(f"  Total in output  : {total_orig + total_aug}")


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

def demo(image_path, n_copies=12):
    """
    Loads one crop and saves a grid showing all augmented versions.
    Also saves individual versions of each major effect for inspection.

    Output: augmentation_demo_manuscript.jpg
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Show individual effects
    img_bgr   = to_color(img)
    effects   = [
        ("original",        to_gray(img_bgr)),
        ("parchment",       to_gray(apply_parchment_background(img_bgr.copy()))),
        ("ink_fading",      to_gray(apply_ink_fading(img_bgr.copy()))),
        ("ink_bleeding",    to_gray(apply_ink_bleeding(img_bgr.copy()))),
        ("elastic",         to_gray(apply_elastic_deform(img_bgr.copy()))),
        ("uneven_light",    to_gray(apply_uneven_illumination(img_bgr.copy()))),
        ("grain_noise",     to_gray(apply_grain_noise(img_bgr.copy()))),
        ("blur",            to_gray(apply_blur(img_bgr.copy()))),
        ("stain",           to_gray(apply_stain(img_bgr.copy()))),
        ("rotation",        to_gray(apply_manuscript_rotation(img_bgr.copy()))),
        ("shear",           to_gray(apply_manuscript_shear(img_bgr.copy()))),
    ]

    # Add combined augmented versions
    aug_versions = augment_manuscript(img, n_copies=n_copies)
    all_versions = effects + [(s, v) for s, v in aug_versions[:n_copies]]

    # Build grid
    cols    = 8
    rows    = (len(all_versions) + cols - 1) // cols
    cell_sz = IMG_SIZE + 22
    grid    = np.ones((rows * cell_sz, cols * cell_sz), np.uint8) * 200

    for idx, (name, v_img) in enumerate(all_versions):
        r  = idx // cols
        c  = idx %  cols
        y0 = r * cell_sz
        x0 = c * cell_sz
        grid[y0:y0+IMG_SIZE, x0:x0+IMG_SIZE] = v_img
        cv2.putText(grid, name[:12],
                    (x0 + 1, y0 + IMG_SIZE + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, 0, 1)

    out_path = "augmentation_demo_manuscript.jpg"
    cv2.imwrite(out_path, grid)
    print(f"Demo saved: {out_path}")
    print(f"Total versions shown: {len(all_versions)}")
    print("Top row = individual effects | Bottom rows = full combined pipeline")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manuscript Newa character augmentation")
    parser.add_argument("--demo",   metavar="IMAGE",
        help="Run demo on a single crop image")
    parser.add_argument("--source", metavar="DIR",
        help="Source folder of labeled manuscript crops")
    parser.add_argument("--output", metavar="DIR",
        help="Output folder for augmented dataset")
    parser.add_argument("--copies", type=int,
        default=COPIES_PER_IMAGE,
        help=f"Augmented copies per original (default {COPIES_PER_IMAGE})")
    parser.add_argument("--overwrite", action="store_true",
        help="Re-generate even if files already exist")
    args = parser.parse_args()

    if args.demo:
        demo(args.demo, n_copies=args.copies)
    elif args.source and args.output:
        augment_folder(args.source, args.output,
                       n_copies=args.copies,
                       overwrite=args.overwrite)
    else:
        print(__doc__)
        print("\nExamples:")
        print("  python augment_manuscript.py --demo ms_crops/ka/ms_0000.png")
        print("  python augment_manuscript.py \\")
        print("      --source dataset_raw/manuscript_data \\")
        print("      --output dataset_aug/manuscript_data \\")
        print("      --copies 20")