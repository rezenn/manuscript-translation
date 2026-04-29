# step4_character_sheets.py
# ─────────────────────────────────────────────────────────────────
# Creates printable handwriting collection sheets.
# Generates TWO sheets:
#   character_sheet_NOTO.pdf    ← Noto Sans reference (you have 7 writers)
#   character_sheet_RANJANA.pdf ← Ranjana reference  (5 more writers)
#
# Layout per row:
#   [REF CHAR] [box1] [box2] [box3] [box4] [box5]
#   shaded                writer fills these
#
# Usage:  python step4_character_sheets.py
# ─────────────────────────────────────────────────────────────────

import os, sys
from PIL import Image, ImageDraw, ImageFont
from utils import setup_utf8
setup_utf8()

sys.path.insert(0, os.path.dirname(__file__))
from newa_classes import NEWA_CHARACTERS, CLASS_TO_RANJANA_KEY

# ── Config ────────────────────────────────────────────────────────
SHEETS = [
    {
        "label":       "NOTO",
        "font_path":   "fonts/NotoSansNewa-Regular.ttf",
        "font_type":   "unicode",
        "ref_size":    52,
        "out":         "character_sheet_NOTO.pdf",
        "description": "Reference font: Noto Sans Newa (Unicode)",
        "writers_note":"For writers who know modern Newa/Prachalit style",
    },
    {
        "label":       "RANJANA",
        "font_path":   "fonts/NithyaRanjanaNU-Regular.otf",
        "font_type":   "unicode",
        "ref_size":    52,
        "out":         "character_sheet_RANJANA.pdf",
        "description": "Reference font: RANJANA (Unicode)",
        "writers_note":"For writers who know modern Newa/Prachalit style",
    },
   
]

# Layout
GRID_COLS   = 4       # characters per row
WRITE_BOXES = 5       # empty boxes writer fills
REF_W       = 90      # reference char box width
BOX_W       = 75      # each writing box width
BOX_H       = 85      # writing box height
COL_W       = REF_W + WRITE_BOXES * (BOX_W + 5) + 25
ROW_H       = 115
H_PADDING   = 25
V_PADDING   = 25
TITLE_H     = 80

# Arial for labels
ARIAL_PATH  = "C:/Windows/Fonts/arial.ttf"
ARIAL_BOLD  = "C:/Windows/Fonts/arialbd.ttf"


def load_label_fonts():
    try:
        title = ImageFont.truetype(ARIAL_BOLD, 16)
        sub   = ImageFont.truetype(ARIAL_PATH, 11)
        tiny  = ImageFont.truetype(ARIAL_PATH, 9)
        return title, sub, tiny
    except Exception:
        d = ImageFont.load_default()
        return d, d, d


def render_ref_char_unicode(draw, char, font_path, size,
                             bx, by, box_w, box_h):
    """Render Noto/Unicode character into reference box."""
    try:
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), char, font=font)
        cw   = bbox[2] - bbox[0]
        ch   = bbox[3] - bbox[1]
        if cw < 3 or ch < 3:
            draw.text((bx+5, by+5), "?", fill='red')
            return
        cx = bx + (box_w - cw) // 2 - bbox[0]
        cy = by + (box_h - ch) // 2 - bbox[1]
        draw.text((cx, cy), char, fill='#000080', font=font)
    except Exception as e:
        draw.text((bx+5, by+5), "!", fill='red')


def render_ref_char_legacy(draw, key_char, font_path,
                            font_size, bx, by, box_w, box_h):
    """
    Render legacy Ranjana character into reference box.
    Auto-scales to ensure visibility.
    """
    if not key_char:
        draw.text((bx+5, by+5), "—", fill='#aaaaaa')
        return

    # Auto-scale: try increasing sizes until glyph is visible
    target_h = int(box_h * 0.65)
    size      = font_size

    for _ in range(15):
        try:
            font = ImageFont.truetype(font_path, size)
            bbox = draw.textbbox((0, 0), key_char, font=font)
            cw   = bbox[2] - bbox[0]
            ch   = bbox[3] - bbox[1]

            if ch >= target_h or size > 200:
                # Center and draw
                cx = bx + (box_w - cw) // 2 - bbox[0]
                cy = by + (box_h - ch) // 2 - bbox[1]
                draw.text((cx, cy), key_char,
                          fill='#000080', font=font)
                return

            # Scale up
            if ch < 3:
                size += 25
            else:
                size = int(size * (target_h / ch)) + 3

        except Exception:
            size += 10

    # Last attempt
    try:
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), key_char, font=font)
        cw, ch = bbox[2]-bbox[0], bbox[3]-bbox[1]
        cx = bx + (box_w - cw) // 2 - bbox[0]
        cy = by + (box_h - ch) // 2 - bbox[1]
        draw.text((cx, cy), key_char, fill='#000080', font=font)
    except Exception:
        pass


def create_sheet(sheet_cfg, newa_characters):
    label       = sheet_cfg["label"]
    font_path   = sheet_cfg["font_path"]
    font_type   = sheet_cfg["font_type"]
    ref_size    = sheet_cfg["ref_size"]
    out_path    = sheet_cfg["out"]

    chars = list(newa_characters.items())
    rows  = (len(chars) + GRID_COLS - 1) // GRID_COLS

    sheet_w = GRID_COLS * COL_W + H_PADDING * 2
    sheet_h = rows * ROW_H + V_PADDING * 2 + TITLE_H

    sheet = Image.new('RGB', (sheet_w, sheet_h), 'white')
    draw  = ImageDraw.Draw(sheet)

    title_font, sub_font, tiny_font = load_label_fonts()

    # ── Title block ───────────────────────────────────────────────
    draw.rectangle([0, 0, sheet_w, TITLE_H],
                   fill='#f8f8f8', outline='#cccccc')

    draw.text((H_PADDING, 10),
              f"Newa Script Handwriting Collection — {label} Style",
              fill='#111111', font=title_font)
    draw.text((H_PADDING, 30),
              sheet_cfg["description"],
              fill='#555555', font=sub_font)
    draw.text((H_PADDING, 45),
              sheet_cfg["writers_note"],
              fill='#888888', font=sub_font)

    # Writer info fields
    rx = sheet_w - 420
    draw.text((rx, 12), "Writer Name:", fill='black', font=sub_font)
    draw.line([(rx+85, 24), (rx+250, 24)], fill='#aaaaaa', width=1)
    draw.text((rx+260, 12), "Date:", fill='black', font=sub_font)
    draw.line([(rx+295, 24), (rx+410, 24)], fill='#aaaaaa', width=1)

    draw.text((rx, 38), "Script Style:", fill='black', font=sub_font)
    draw.text((rx+78, 38), label, fill='#0000cc', font=sub_font)

    draw.text((H_PADDING, 62),
              f"Instructions: Copy each character from the grey reference box "
              f"into the 5 empty boxes beside it. Total: {len(chars)} characters.",
              fill='#333333', font=tiny_font)

    # ── Column headers ────────────────────────────────────────────
    header_y = TITLE_H + 5
    for col in range(GRID_COLS):
        bx = H_PADDING + col * COL_W
        draw.text((bx + 5, header_y),
                  "Ref", fill='#999999', font=tiny_font)
        for b in range(WRITE_BOXES):
            wx = bx + REF_W + 10 + b * (BOX_W + 5)
            draw.text((wx + BOX_W//2 - 5, header_y),
                      str(b+1), fill='#bbbbbb', font=tiny_font)

    # ── Character rows ────────────────────────────────────────────
    missing_chars = []

    for i, (class_name, char_unicode) in enumerate(chars):
        col   = i % GRID_COLS
        row   = i // GRID_COLS
        base_x = H_PADDING + col * COL_W
        base_y = TITLE_H + 22 + V_PADDING + row * ROW_H

        # Reference box
        draw.rectangle(
            [base_x, base_y,
             base_x + REF_W, base_y + BOX_H],
            fill='#eef2ff', outline='#8899cc', width=2)

        if font_type == "unicode":
            render_ref_char_unicode(
                draw, char_unicode, font_path, ref_size,
                base_x, base_y, REF_W, BOX_H)
        else:
            key_char = CLASS_TO_RANJANA_KEY.get(class_name)
            if not key_char:
                missing_chars.append(class_name)
            render_ref_char_legacy(
                draw, key_char, font_path, ref_size,
                base_x, base_y, REF_W, BOX_H)

        # Class label under reference box
        draw.text((base_x + 2, base_y + BOX_H + 2),
                  class_name[:12],
                  fill='#666666', font=tiny_font)

        # Number label
        draw.text((base_x + 2, base_y + 2),
                  f"#{i+1}", fill='#aaaaaa', font=tiny_font)

        # Writing boxes
        for b in range(WRITE_BOXES):
            wx = base_x + REF_W + 10 + b * (BOX_W + 5)
            wy = base_y + (BOX_H - BOX_H) // 2

            draw.rectangle(
                [wx, wy, wx + BOX_W, wy + BOX_H],
                fill='white', outline='#cccccc', width=1)

            # Faint guide cross
            mid_x = wx + BOX_W // 2
            mid_y = wy + BOX_H // 2
            draw.line([(mid_x-8, mid_y), (mid_x+8, mid_y)],
                      fill='#eeeeee', width=1)
            draw.line([(mid_x, mid_y-8), (mid_x, mid_y+8)],
                      fill='#eeeeee', width=1)

    # ── Save ──────────────────────────────────────────────────────
    sheet.save(out_path, "PDF", resolution=150)
    print(f"  ✓ Saved: {out_path}")
    print(f"    Sheet: {sheet_w}×{sheet_h}px  |  "
          f"{len(chars)} chars  |  "
          f"{len(chars)*WRITE_BOXES} samples/writer")

    if missing_chars:
        print(f"  ⚠ {len(missing_chars)} classes have no Ranjana key "
              f"mapping (will show blank ref): "
              f"{missing_chars[:5]}...")

    return out_path


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 4 — HANDWRITING COLLECTION SHEETS")
    print("=" * 60)

    for sheet_cfg in SHEETS:
        label = sheet_cfg["label"]
        print(f"\n[{label}]")

        if not os.path.exists(sheet_cfg["font_path"]):
            print(f"  ✗ Font not found: {sheet_cfg['font_path']}")
            print(f"    Sheet cannot be created — add font first.")
            continue

        create_sheet(sheet_cfg, NEWA_CHARACTERS)

    print(f"\n{'='*60}")
    print("Sheets saved. Print at A3 size for comfortable writing.")
    print()
    print("Collection plan:")
    print("  NOTO sheet   → 7 writers already done ✓")
    print("  RANJANA sheet → 5 more writers (calligraphic style)")
    print()
    print("After collecting, run:  python step5_process_handwritten.py")