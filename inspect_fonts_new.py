# step1_inspect_fonts.py
# ─────────────────────────────────────────────────────────────────
# Run this FIRST after downloading your fonts.
# It tells you:
#   - Is each font Unicode or Legacy?
#   - For Legacy fonts: renders ALL keys so you can see which
#     keyboard key produces which Newa character.
#
# Usage:  python step1_inspect_fonts.py
# ─────────────────────────────────────────────────────────────────

import os
import sys
from PIL import Image, ImageDraw, ImageFont
from utils import setup_utf8
setup_utf8()
# ── Configure your font paths here ───────────────────────────────
FONTS = {
    "noto_sans":  "fonts/NotoSansNewa-Regular.ttf",   # Unicode
    "ranjana":  "fonts/NithyaRanjanaNU.otf",   # Unicode

}

OUTPUT_DIR = "font_inspection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1. Detect font type using fonttools
# ─────────────────────────────────────────────────────────────────
def detect_font_type(font_path):
    try:
        from fonttools.ttLib import TTFont
        tt   = TTFont(font_path)
        cmap = tt.getBestCmap()
        if cmap is None:
            return "UNKNOWN", set(), set()

        cps   = set(cmap.keys())
        newa  = {cp for cp in cps if 0x11400 <= cp <= 0x1147F}
        ascii_cps = {cp for cp in cps if 0x20 <= cp <= 0x7E}

        if len(newa) > 5:
            return "UNICODE", newa, ascii_cps
        elif len(ascii_cps) > 20:
            return "LEGACY",  newa, ascii_cps
        else:
            return "UNKNOWN", newa, ascii_cps
    except Exception as e:
        return "UNKNOWN", set(), set()


# ─────────────────────────────────────────────────────────────────
# 3. For UNICODE fonts: render the actual Newa characters
# ─────────────────────────────────────────────────────────────────
def render_unicode_sample(font_path, font_id, newa_characters):
    COLS   = 10
    CELL   = 80
    chars  = list(newa_characters.items())
    ROWS   = (len(chars) + COLS - 1) // COLS
    W      = COLS * CELL + 20
    H      = ROWS * CELL + 60

    sheet  = Image.new('RGB', (W, H), 'white')
    draw   = ImageDraw.Draw(sheet)

    try:
        newa_font  = ImageFont.truetype(font_path, 44)
        label_font = ImageFont.truetype(
            "C:/Windows/Fonts/arial.ttf", 9)
    except Exception:
        newa_font  = ImageFont.load_default()
        label_font = newa_font

    draw.text((10, 8),
              f"Unicode render check — {font_id}",
              fill='black')

    for i, (cname, char) in enumerate(chars):
        col  = i % COLS
        row  = i // COLS
        bx   = 10 + col * CELL
        by   = 35 + row * CELL

        draw.rectangle([bx, by, bx + CELL - 3, by + CELL - 3],
                       outline='#bbbbbb', width=1)

        try:
            bbox = draw.textbbox((0, 0), char, font=newa_font)
            gw, gh = bbox[2]-bbox[0], bbox[3]-bbox[1]
            if gw < 3 or gh < 3:
                draw.text((bx+5, by+5), "?",
                          fill='red', font=label_font)
            else:
                gx = bx + (CELL-gw)//2 - bbox[0]
                gy = by + (CELL-gh)//2 - bbox[1]
                draw.text((gx, gy), char,
                          fill='#000080', font=newa_font)
        except Exception:
            pass

        draw.text((bx+2, by+CELL-13),
                  cname[:8], fill='#888888', font=label_font)

    out_path = os.path.join(OUTPUT_DIR,
                            f"{font_id}_unicode_check.png")
    sheet.save(out_path)
    print(f"  ✓ Unicode check sheet: {out_path}")


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from newa_classes import NEWA_CHARACTERS

    print("=" * 60)
    print("STEP 1 — FONT INSPECTION")
    print("=" * 60)

    for font_id, font_path in FONTS.items():
        print(f"\n[{font_id}]")

        if not os.path.exists(font_path):
            print(f"  ✗ File not found: {font_path}")
            print(f"    → Download and place at: {font_path}")
            continue

        ftype, newa_cps, ascii_cps = detect_font_type(font_path)
        print(f"  Type detected : {ftype}")
        print(f"  Newa glyphs   : {len(newa_cps)}")
        print(f"  ASCII glyphs  : {len(ascii_cps)}")

        if ftype == "UNICODE":
            print(f"  → Use with Unicode codepoints directly ✓")
            render_unicode_sample(font_path, font_id,
                                  NEWA_CHARACTERS)
     

    print(f"\nDone. Check: {OUTPUT_DIR}/")
    print("IMPORTANT: Open the *_key_mapping.png files")
    print("and verify RANJANA_KEY_MAP in newa_classes.py matches!")