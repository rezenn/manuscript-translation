"""
generate_compound_classes.py
─────────────────────────────────────────────────────────────────────
Generates consonant+matra compound classes (e.g. 'ma_aa' = म+ा fused)
FIX: Forces a HUGE temporary canvas for RAQM so HarfBuzz doesn't 
disable complex shaping due to small font sizes. 
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import setup_utf8
setup_utf8()

sys.path.insert(0, os.path.dirname(__file__))
from newa_classes import NEWA_CHARACTERS

# Use a larger image size because small canvases break RAQM shaping
IMG_SIZE = 128


def render_char_centered(text: str, font: ImageFont.FreeTypeFont, img_size: int = IMG_SIZE):
    """
    Renders `text` with the GIVEN font object.
    
    CRITICAL FIX: We use a MASSIVE canvas (at least 2000x2000) to render 
    the text first. This guarantees HarfBuzz/RAQM does not fall back to 
    simple layout. We then crop and resize down to the requested size.
    """
    # CRITICAL FIX: Use a canvas so large it forces RAQM to stay active
    render_canvas_size = 2000  
    pad = 200
    
    # Create a giant temporary canvas
    tmp = Image.new("L", (render_canvas_size, render_canvas_size), 255)
    draw = ImageDraw.Draw(tmp)
    
    # Get the bounding box
    bbox = draw.textbbox((pad, pad), text, font=font)
    
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
        
    # Draw the text on the giant canvas
    draw.text((pad, pad), text, font=font, fill=0)

    # Crop exactly around the rendered glyph
    crop = tmp.crop(bbox)
    w, h = crop.size
    if w == 0 or h == 0:
        return None

    # Resize to target size while keeping aspect ratio
    scale = (img_size - 12) / max(w, h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    crop = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Paste onto final white canvas
    canvas = Image.new("L", (img_size, img_size), 255)
    canvas.paste(crop, ((img_size - new_w) // 2, (img_size - new_h) // 2))
    return canvas


def is_blank(img, ink_thresh=200, min_ink_px=15) -> bool:
    arr = np.array(img)
    return int((arr < ink_thresh).sum()) < min_ink_px


CONSONANTS = [
    'ka', 'kha', 'ga', 'gha', 'nga', 'ca', 'cha', 'ja', 'jha', 'nya',
    'tta', 'ttha', 'dda', 'ddha', 'nna', 'ta', 'tha', 'da', 'dha', 'na',
    'pa', 'pha', 'ba', 'bha', 'ma', 'ya', 'ra', 'la', 'wa', 'sha',
    'ssa', 'sa', 'ha',
]

MATRAS = [
    'matra_aa', 'matra_i', 'matra_ii', 'matra_u', 'matra_uu',
    'matra_e', 'matra_ai', 'matra_o', 'matra_au',
]

MATRA_SUFFIX = {
    'matra_aa': 'aa', 'matra_i': 'i', 'matra_ii': 'ii',
    'matra_u': 'u', 'matra_uu': 'uu', 'matra_e': 'e',
    'matra_ai': 'ai', 'matra_o': 'o', 'matra_au': 'au',
}

FONTS = {
    "noto_sans": {
        "path": "fonts/NotoSansNewa-Regular.ttf",
        "sizes": [36, 40, 44, 48, 52, 56, 60],
        "out": "dataset_raw/synthetic_compound_noto",
    },
    "Ranjana": {
        "path": "fonts/NithyaRanjanaNU-Regular.otf",
        "sizes": [36, 40, 48, 56, 64, 72, 80, 88],
        "out": "dataset_raw/synthetic_compound_ranjana",
    },
}

EXTRA_LOW_SIGNAL_SIZES = {
    "noto_sans": [32, 38, 42, 46, 50, 54, 58, 64, 70],
    "Ranjana":   [32, 38, 44, 52, 60, 68, 76, 84, 92, 100],
}

LOW_SIGNAL_THRESHOLD = 800
REFERENCE_FONT_SIZE = 100
REFERENCE_CANVAS = 220


def build_compound_classes() -> dict:
    compounds = {}
    for cons in CONSONANTS:
        cons_char = NEWA_CHARACTERS.get(cons)
        if not cons_char:
            continue
        for matra in MATRAS:
            matra_char = NEWA_CHARACTERS.get(matra)
            if not matra_char:
                continue
            class_name = f"{cons}_{MATRA_SUFFIX[matra]}"
            compounds[class_name] = (cons, matra, cons_char + matra_char)
    return compounds


def _render_to_array(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    # Also use a large temporary canvas here for RAQM
    img = Image.new("L", (REFERENCE_CANVAS, REFERENCE_CANVAS), 255)
    ImageDraw.Draw(img).text((10, 10), text, font=font, fill=0)
    return np.array(img)


def detect_low_signal_classes(compound_classes: dict, font_path: str) -> dict:
    font = ImageFont.truetype(
        font_path, REFERENCE_FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    diffs = {}
    bare_cache = {}
    for class_name, (cons, matra, char_seq) in compound_classes.items():
        if cons not in bare_cache:
            bare_cache[cons] = _render_to_array(NEWA_CHARACTERS[cons], font)
        bare_arr = bare_cache[cons]
        comp_arr = _render_to_array(char_seq, font)
        diffs[class_name] = int((bare_arr != comp_arr).sum())
    return diffs


def generate(compound_classes: dict, low_signal: set):
    total_saved = total_skipped = 0

    for font_label, font_cfg in FONTS.items():
        font_path = font_cfg["path"]
        base_sizes = font_cfg["sizes"]
        extra_sizes = EXTRA_LOW_SIGNAL_SIZES[font_label]
        out_base = font_cfg["out"]
        os.makedirs(out_base, exist_ok=True)
        saved = skipped = 0

        for class_name, (cons, matra, char_seq) in tqdm(
                compound_classes.items(),
                desc=f"{font_label} compounds"):

            class_dir = os.path.join(out_base, class_name)
            os.makedirs(class_dir, exist_ok=True)

            sizes = extra_sizes if class_name in low_signal else base_sizes

            for size in sizes:
                try:
                    # Load font with RAQM
                    font = ImageFont.truetype(
                        font_path, size, layout_engine=ImageFont.Layout.RAQM)
                    
                    # CRITICAL FIX: We don't need to pass a large img_size here
                    # because the function internally uses a fixed 2000x2000 canvas
                    # to guarantee RAQM engagement.
                    img = render_char_centered(char_seq, font, img_size=IMG_SIZE)
                    
                    if img is not None and not is_blank(img):
                        img.save(os.path.join(
                            class_dir, f"{font_label}_{size}px.png"))
                        saved += 1
                    else:
                        skipped += 1
                except Exception as e:
                    # Print error for debugging
                    print(f"Error processing {class_name} with size {size}: {e}")
                    skipped += 1

        print(f"  {font_label}: saved={saved}  skipped={skipped}  -> {out_base}/")
        total_saved += saved
        total_skipped += skipped

    print(f"\nTotal compound images saved:   {total_saved}")
    print(f"Total compound images skipped: {total_skipped}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating consonant+matra compound classes (with CRITICAL RAQM fix)")
    print("=" * 60)

    compound_classes = build_compound_classes()
    print(f"Built {len(compound_classes)} compound class definitions")
    print()

    print("Checking which compounds are visually subtle in this font...")
    diffs = detect_low_signal_classes(compound_classes, FONTS["Ranjana"]["path"])
    low_signal = {name for name, d in diffs.items() if d < LOW_SIGNAL_THRESHOLD}
    print(f"  {len(low_signal)}/{len(compound_classes)} flagged low-signal "
          f"(not a bug -- some matras are genuinely subtle in this font)")
    print()

    generate(compound_classes, low_signal)

    print()
    print("DONE. Run augmentated_data.py and build_data.py next.")