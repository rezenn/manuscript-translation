
import os, sys
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utils import setup_utf8
setup_utf8()

sys.path.insert(0, os.path.dirname(__file__))
from newa_classes import (NEWA_CHARACTERS,
                           CLASS_TO_RANJANA_KEY)
 
# ── Settings ─────────────────────────────────────────────────────
IMG_SIZE        = 128
PADDING_FRAC    = 0.12   # 12% padding around character
MIN_FILL_FRAC   = 0.25   # glyph must fill at least 25% of canvas
TARGET_FILL     = 0.65   # aim for glyph to fill ~65% of canvas
 
NOTO_SIZES      = [36, 40, 44, 48, 52, 56, 60]   # 7 images/class
RANJANA_SIZES   = [40, 48, 56, 64, 72, 80, 88]   # start bigger for legacy
 
FONTS = {
    "noto_sans": {
        "path": "fonts/NotoSansNewa-Regular.ttf",
        "type":  "unicode",
        "sizes": NOTO_SIZES,
        "out":   "dataset_raw/synthetic_noto",
    },
    "Ranjana": {
        "path": "fonts/NithyaRanjanaNU-Regular.otf",
        "type":  "unicode",
        "sizes": NOTO_SIZES,
        "out":   "dataset_raw/synthetic_ranjana",
    },
}
 
 
# ─────────────────────────────────────────────────────────────────
# Core rendering functions
# ─────────────────────────────────────────────────────────────────
 
def get_glyph_size(draw, char, font):
    """Returns actual rendered pixel size of a character."""
    bbox = draw.textbbox((0, 0), char, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h
 
 
def auto_scale_font(font_path, char, target_px, start_size=40):
    """
    THE KEY FIX for legacy fonts.
    Increases font size until glyph actually fills target_px pixels.
    Legacy fonts (Ranjana) often render tiny at small pt sizes
    because their internal metrics are different.
    """
    img  = Image.new('L', (512, 512), 255)
    draw = ImageDraw.Draw(img)
 
    size = start_size
    for _ in range(80):   # max 80 attempts
        try:
            font    = ImageFont.truetype(font_path, size)
            w, h    = get_glyph_size(draw, char, font)
            current = max(w, h)
 
            if current >= target_px:
                return font, size
            if current < 3:
                # Glyph not rendering at all — jump up fast
                size += 20
            else:
                # Scale proportionally
                ratio = target_px / current
                size  = int(size * ratio) + 2
        except Exception:
            size += 5
 
    # Return best we got
    return ImageFont.truetype(font_path, size), size
 
 
def render_char_centered(char, font, img_size=IMG_SIZE,
                          padding_frac=PADDING_FRAC):
    """
    Renders a single character centered on a white canvas.
    Works for both Unicode and legacy fonts.
    """
    img  = Image.new('L', (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
 
    bbox = draw.textbbox((0, 0), char, font=font)
    w    = bbox[2] - bbox[0]
    h    = bbox[3] - bbox[1]
 
    if w < 3 or h < 3:
        return None   # blank glyph
 
    pad  = int(img_size * padding_frac)
    avail = img_size - 2 * pad
 
    # Scale down if too big
    scale = min(avail / w, avail / h, 1.0)
    if scale < 1.0:
        new_size = int(font.size * scale)
        try:
            font = ImageFont.truetype(font.path, new_size)
        except Exception:
            pass
        bbox = draw.textbbox((0, 0), char, font=font)
        w    = bbox[2] - bbox[0]
        h    = bbox[3] - bbox[1]
 
    # Center
    x = (img_size - w) // 2 - bbox[0]
    y = (img_size - h) // 2 - bbox[1]
 
    draw.text((x, y), char, fill=0, font=font)
    return img
 
 
def is_blank(img, threshold=250, min_dark_pixels=10):
    """Returns True if image is empty (no actual glyph rendered)."""
    import numpy as np
    arr = np.array(img)
    return (arr < threshold).sum() < min_dark_pixels
 
 
# ─────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────
 
def generate_unicode_font(font_cfg, newa_characters):
    font_path = font_cfg["path"]
    out_base  = font_cfg["out"]
    sizes     = font_cfg["sizes"]
 
    os.makedirs(out_base, exist_ok=True)
    saved = skipped = 0
 
    for class_name, char_unicode in tqdm(
            newa_characters.items(),
            desc=f"Noto Unicode"):
 
        class_dir = os.path.join(out_base, class_name)
        os.makedirs(class_dir, exist_ok=True)
 
        for size in sizes:
            try:
                font = ImageFont.truetype(font_path, size)
                img  = render_char_centered(char_unicode, font)
                if img and not is_blank(img):
                    img.save(os.path.join(class_dir,
                             f"noto_{size}px.png"))
                    saved += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
 
    print(f"  Noto: saved={saved}  skipped={skipped}")
    return saved
 
 
def generate_legacy_font(font_cfg, newa_characters,
                          class_to_key, font_label):
    font_path = font_cfg["path"]
    out_base  = font_cfg["out"]
    sizes     = font_cfg["sizes"]
 
    os.makedirs(out_base, exist_ok=True)
    saved = skipped = no_key = 0
 
    # Pre-compute auto-scaled font sizes for this font
    # (legacy fonts need much larger pt sizes to render visibly)
    target_px = int(IMG_SIZE * TARGET_FILL)
 
    for class_name, char_unicode in tqdm(
            newa_characters.items(),
            desc=f"Ranjana Legacy"):
 
        key_char = class_to_key.get(class_name)
        if not key_char:
            no_key += 1
            continue
 
        class_dir = os.path.join(out_base, class_name)
        os.makedirs(class_dir, exist_ok=True)
 
        # Find a good starting font size for this key/glyph
        # (some glyphs render at different natural sizes)
        base_font, base_size = auto_scale_font(
            font_path, key_char, target_px,
            start_size=sizes[0])
 
        # Generate multiple sizes around the base
        size_variants = [
            max(20, base_size + offset)
            for offset in [-16, -8, 0, 8, 16, 24, 32]
        ]
 
        for i, size in enumerate(size_variants):
            try:
                font = ImageFont.truetype(font_path, size)
                img  = render_char_centered(key_char, font)
                if img and not is_blank(img):
                    img.save(os.path.join(class_dir,
                             f"{font_label}_{i:02d}_{size}px.png"))
                    saved += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1
 
    print(f"  {font_label}: saved={saved}  "
          f"skipped={skipped}  no_key={no_key}")
    print(f"  Note: {no_key} classes have no key mapping in this font")
    return saved
 
 
# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from newa_classes import NEWA_CHARACTERS, CLASS_TO_RANJANA_KEY
 
    print("=" * 60)
    print("STEP 2 — SYNTHETIC DATA GENERATION")
    print("=" * 60)
 
    total = 0
 
    for font_id, font_cfg in FONTS.items():
        print(f"\n[{font_id}]")
 
        if not os.path.exists(font_cfg["path"]):
            print(f"  ✗ Not found: {font_cfg['path']} — skipping")
            continue
 
        if font_cfg["type"] == "unicode":
            n = generate_unicode_font(font_cfg, NEWA_CHARACTERS)
        else:
            n = generate_legacy_font(
                font_cfg, NEWA_CHARACTERS,
                CLASS_TO_RANJANA_KEY, font_id)
        total += n
 
    print(f"\n{'='*60}")
    print(f"Total synthetic images: {total}")
    per_class = total // len(NEWA_CHARACTERS) if NEWA_CHARACTERS else 0
    print(f"Approx per class: {per_class}")
    print(f"\nOutput folders:")
    for fid, cfg in FONTS.items():
        print(f"  {cfg['out']}/")