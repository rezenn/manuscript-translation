"""
step5a_process_sheet.py  —  Newa Script OCR  (FIXED v2)
============================================================
WHAT WAS WRONG (and what was fixed):

  BUG 1 — Wrong GRID_COLS
    Before : GRID_COLS = 5
    After  : GRID_COLS = 8
    Why    : Your sheet has 8 character columns per row, not 5.
             With the wrong value every group of cells was offset
             so the ref box landed on writing boxes and vice versa.

  BUG 2 — Wrong ref_w fraction
    Before : ref_w = int(col_w * 0.22)   ← 22% of column
    After  : ref_w = int(col_w * 0.30)   ← 30% of column
    Why    : Pixel analysis of the debug image showed the reference
             box occupies 38px out of every 125px group = 30.4%.
             At 22% the script was cutting into the writing boxes
             instead of skipping the reference character.

  BUG 3 — Wrong title_frac
    Before : title_frac = 0.06
    After  : title_frac = 0.05
    Why    : The title area is slightly smaller than assumed,
             causing the first row of cells to be missed or shifted.

  BUG 4 — Wrong row_h calculation
    Before : box_h = int(row_h * 0.70)
    After  : box_h = int(row_h * 0.60)
    Why    : Each character row has a sub-label row beneath it
             (showing the class name). The char area is only ~60%
             of the total row height. At 70% the crops were
             bleeding into the label text below.

USAGE:
  python step5a_process_sheet.py

Edit the JOBS section at the bottom to point to your photos.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


_this_dir = os.path.dirname(os.path.abspath(__file__)) \
            if '__file__' in dir() else os.getcwd()
sys.path.insert(0, _this_dir)

try:
    from newa_classes import NEWA_CHARACTERS
except ImportError:
    print("⚠  Could not import newa_classes — using placeholder set")
    NEWA_CHARACTERS = {f"char_{i:02d}": i for i in range(82)}

# ═══════════════════════════════════════════════════════════════
# ▼▼▼  CORRECTED CONFIGURATION  ▼▼▼
# ═══════════════════════════════════════════════════════════════

IMG_SIZE    = 128
WRITE_BOXES = 5       # writing boxes per character
GRID_COLS   = 8       # ← FIXED: was 5, your sheet has 8 columns
DEBUG_DIR   = "debug_crops"
AUGMENT     = True

# Light augmentation settings
AUG_ROTATIONS  = [-7, -3, 3, 7]
AUG_BRIGHTNESS = [0.85, 1.15]


# ═══════════════════════════════════════════════════════════════
# PERSPECTIVE CORRECTION
# ═══════════════════════════════════════════════════════════════

def deskew_sheet(img):
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  ⚠ No contours — skipping deskew")
        return img
    img_area = img.shape[0] * img.shape[1]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.10:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx, area
    if best is None:
        print("  ⚠ Sheet boundary not found — using full image")
        return img
    pts  = best.reshape(4, 2).astype(np.float32)
    rect = _order_corners(pts)
    W, H = 2480, 3508
    dst  = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)
    M    = cv2.getPerspectiveTransform(rect, dst)
    out  = cv2.warpPerspective(img, M, (W, H))
    print("  ✓ Perspective correction applied")
    return out


def _order_corners(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ═══════════════════════════════════════════════════════════════
# GEOMETRY-BASED CELL EXTRACTION  (FIXED)
# ═══════════════════════════════════════════════════════════════

def extract_cells_by_geometry(img_gray, n_chars,
                               grid_cols=GRID_COLS,
                               write_boxes=WRITE_BOXES,
                               title_frac=0.05):   # ← FIXED: was 0.06
    """
    Computes the bounding box of every writing cell directly from
    the known sheet geometry.

    FIXED PARAMETERS (measured from pixel analysis of debug image):
      grid_cols  = 8      (was 5)
      ref_frac   = 0.30   (was 0.22) — ref box is 30% of column width
      title_frac = 0.05   (was 0.06)
      box_h_frac = 0.60   (was 0.70) — char area is 60% of row height

    Returns: list of lists — cells[char_idx] = [(x,y,w,h), ...]
    """
    h, w  = img_gray.shape
    rows  = (n_chars + grid_cols - 1) // grid_cols

    # Margins
    margin_x = int(w * 0.015)
    margin_y = int(h * 0.015)
    title_h  = int(h * title_frac)

    content_w = w - 2 * margin_x
    content_h = h - 2 * margin_y - title_h

    col_w = content_w // grid_cols
    row_h = content_h // rows

    # ── KEY FIX ─────────────────────────────────────────────────
    # ref_w = 30% of col_w  (pixel analysis showed 38/125 = 30.4%)
    # Previously this was 22% which caused the ref box to overlap
    # with the first writing box.
    ref_w   = int(col_w * 0.30)      # ← FIXED: was 0.22

    box_gap = max(2, col_w // 50)
    avail   = col_w - ref_w - 8
    box_w   = max(8, (avail - box_gap * (write_boxes - 1)) // write_boxes)

    # ── KEY FIX ─────────────────────────────────────────────────
    # box_h = 60% of row_h  (each row has char area + label area below)
    # Previously this was 70% which bled into the label text
    box_h   = max(8, int(row_h * 0.60))   # ← FIXED: was 0.70

    all_cells = []

    for i in range(n_chars):
        col = i % grid_cols
        row = i // grid_cols

        base_x = margin_x + col * col_w
        base_y = margin_y + title_h + row * row_h

        char_cells = []
        for b in range(write_boxes):
            bx = base_x + ref_w + 8 + b * (box_w + box_gap)
            by = base_y + int(row_h * 0.05)   # small top pad
            bx = min(bx, w - box_w - 1)
            by = min(by, h - box_h - 1)
            char_cells.append((bx, by, box_w, box_h))

        all_cells.append(char_cells)

    return all_cells


# ═══════════════════════════════════════════════════════════════
# GRID LINE DETECTION  (unchanged — used as primary method)
# ═══════════════════════════════════════════════════════════════

def find_grid_lines(img_gray):
    h, w = img_gray.shape
    _, binary = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    row_darkness = (binary > 0).sum(axis=1).astype(float) / w
    col_darkness = (binary > 0).sum(axis=0).astype(float) / h
    h_mask = row_darkness > 0.30
    v_mask = col_darkness > 0.30
    h_positions = _find_line_centers(h_mask)
    v_positions = _find_line_centers(v_mask)
    print(f"  Grid detection: {len(h_positions)} h-lines, "
          f"{len(v_positions)} v-lines found")
    return h_positions, v_positions


def _find_line_centers(mask):
    centers = []
    in_band, start = False, 0
    for i, val in enumerate(mask):
        if val and not in_band:
            in_band, start = True, i
        elif not val and in_band:
            centers.append((start + i) // 2)
            in_band = False
    if in_band:
        centers.append((start + len(mask)) // 2)
    return centers


def get_cells(img_gray, n_chars, write_boxes, grid_cols):
    """Try grid line detection first, fall back to geometry."""
    h_lines, v_lines = find_grid_lines(img_gray)

    # Need enough lines to be useful
    # With 8 cols and 5 write+1 ref = 6 slots + separators
    # expect at least grid_cols * 4 vertical lines
    if len(v_lines) >= grid_cols * 4 and len(h_lines) >= 4:
        cells = _cells_from_lines(h_lines, v_lines, n_chars,
                                   write_boxes, grid_cols)
        if cells and len(cells) >= n_chars * 0.5:
            print(f"  ✓ Grid detection: {len(cells)} chars found")
            return cells

    print("  → Geometry fallback")
    return extract_cells_by_geometry(img_gray, n_chars,
                                      grid_cols, write_boxes)


def _cells_from_lines(h_lines, v_lines, n_chars,
                       write_boxes, grid_cols):
    """Extract cells from detected grid line positions."""
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    all_cells = []
    # Group v_lines into character groups
    # Each group: [ref_left, ref_right, w1_right, w2_right, ...]
    n_slots = write_boxes + 1   # ref + write boxes
    group_step = max(1, len(v_lines) // grid_cols)

    row_i = 0
    while row_i < len(h_lines) - 1:
        y0 = h_lines[row_i]
        y1 = h_lines[row_i + 1]
        ch = y1 - y0
        if ch < 15:
            row_i += 1
            continue

        for col in range(grid_cols):
            base_v = col * group_step
            if base_v + n_slots >= len(v_lines):
                break
            # Skip ref box (first slot), use writing boxes
            for b in range(write_boxes):
                vi  = base_v + 1 + b
                vi2 = vi + 1
                if vi2 >= len(v_lines):
                    break
                x0 = v_lines[vi]
                x1 = v_lines[vi2]
                cw = x1 - x0
                if cw < 8:
                    continue
                if len(all_cells) <= col + row_i * grid_cols:
                    all_cells.append([])
                while len(all_cells) <= col + row_i * grid_cols:
                    all_cells.append([])
                all_cells[col + row_i * grid_cols].append(
                    (x0, y0, cw, ch))
        row_i += 1

    return all_cells if all_cells else None


# ═══════════════════════════════════════════════════════════════
# CROP EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_crop(img_gray, x, y, w, h, pad=5):
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(img_gray.shape[1], x + w - pad)
    y1 = min(img_gray.shape[0], y + h - pad)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img_gray[y0:y1, x0:x1].copy()
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
    if np.median(crop) < 128:
        crop = cv2.bitwise_not(crop)
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                      interpolation=cv2.INTER_AREA)


def is_blank(img, dark_thresh=200, min_dark_px=20):
    if img is None:
        return True
    return (img < dark_thresh).sum() < min_dark_px


# ═══════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def augment_crop(img):
    results = []
    h, w    = img.shape
    center  = (w // 2, h // 2)
    for deg in AUG_ROTATIONS:
        M   = cv2.getRotationMatrix2D(center, deg, 1.0)
        rot = cv2.warpAffine(img, M, (w, h),
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=255)
        results.append((f"r{deg:+d}", rot))
    for factor in AUG_BRIGHTNESS:
        bright = np.clip(img.astype(float) * factor,
                         0, 255).astype(np.uint8)
        results.append((f"b{int(factor*100)}", bright))
    return results


# ═══════════════════════════════════════════════════════════════
# STORAGE
# ═══════════════════════════════════════════════════════════════

def save_crop(img, out_dir, class_name, writer_id, style,
              suffix="", existing_count=0, idx=0):
    class_dir = os.path.join(out_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    fname = f"hw_{style}_w{writer_id}_{existing_count + idx:04d}"
    if suffix:
        fname += f"_{suffix}"
    fname += ".png"
    cv2.imwrite(os.path.join(class_dir, fname), img)
    return os.path.join(class_dir, fname)


def count_existing(out_dir, class_name):
    d = os.path.join(out_dir, class_name)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.png')])


# ═══════════════════════════════════════════════════════════════
# DEBUG VISUALISER — shows what the script "thinks" is each cell
# ═══════════════════════════════════════════════════════════════

def save_geometry_preview(img_gray, all_cells, image_path,
                           chars, grid_cols):
    """
    Saves a colour image overlaying the computed cell boxes
    onto the sheet, WITHOUT actually processing anything.

    Use this to visually verify the geometry is correct
    BEFORE running a full extraction.

    Green  = writing cell box (what will be cropped)
    Blue   = ref box position (will be skipped)
    Red    = out-of-bounds region
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h, w  = img_gray.shape

    n_chars   = len(chars)
    grid_cols_ = grid_cols
    rows       = (n_chars + grid_cols_ - 1) // grid_cols_

    # Also draw the ref box for each character (blue)
    margin_x = int(w * 0.015)
    margin_y = int(h * 0.015)
    title_h  = int(h * 0.05)
    col_w    = (w - 2*margin_x) // grid_cols_
    row_h    = (h - 2*margin_y - title_h) // rows
    ref_w    = int(col_w * 0.30)

    for i in range(n_chars):
        col = i % grid_cols_
        row = i // grid_cols_
        rx  = margin_x + col * col_w
        ry  = margin_y + title_h + row * row_h
        # Draw ref box outline in blue
        cv2.rectangle(debug, (rx, ry),
                      (rx + ref_w, ry + row_h - 4),
                      (200, 100, 0), 1)

    # Draw writing cell boxes
    for i, (class_name, _) in enumerate(chars):
        if i >= len(all_cells):
            break
        for x, y, cw, ch in all_cells[i]:
            color = (0, 200, 0)   # green = valid
            if x < 0 or y < 0 or x+cw > w or y+ch > h:
                color = (0, 0, 200)   # red = out of bounds
            cv2.rectangle(debug, (x, y), (x+cw, y+ch), color, 1)
            cv2.putText(debug, class_name[:4],
                        (x+1, y+10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.22, (200, 80, 0), 1)

    stem  = Path(image_path).stem
    dpath = os.path.join(DEBUG_DIR, f"{stem}_geometry_preview.jpg")
    # Scale down if large
    dh, dw = debug.shape[:2]
    if dw > 1800:
        scale = 1800 / dw
        debug = cv2.resize(debug, (1800, int(dh * scale)))
    cv2.imwrite(dpath, debug)
    print(f"\n  Geometry preview: {dpath}")
    print(f"  Green = writing boxes (will be cropped)")
    print(f"  Blue  = reference box position (skipped)")
    print(f"  Open this image to verify boxes align with your writing!")
    return dpath


# ═══════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════

def process_sheet(
    image_path  : str,
    output_dir  : str,
    writer_id,
    style       : str  = "noto",
    deskew      : bool = True,
    augment     : bool = AUGMENT,
    debug       : bool = True,
    preview_only: bool = False,   # ← NEW: just show geometry, don't save crops
    write_boxes : int  = WRITE_BOXES,
    grid_cols   : int  = GRID_COLS,
):
    """
    Process one handwriting sheet photo.

    NEW OPTION — preview_only=True:
      Saves a geometry preview image showing where cells will be
      cropped, without actually extracting anything.
      Use this first to verify the geometry is correct!

      python step5a_process_sheet.py  (with PREVIEW_ONLY=True below)
      → Open debug_crops/<name>_geometry_preview.jpg
      → If boxes are wrong, adjust GRID_COLS or ref_frac
      → Then set PREVIEW_ONLY=False and run for real
    """
    print(f"\n{'='*60}")
    print(f"Sheet: {image_path}")
    print(f"Writer: {writer_id}   Style: {style}")
    if preview_only:
        print(f"MODE: PREVIEW ONLY (no crops saved)")
    print(f"{'='*60}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"  ✗ Cannot read: {image_path}")
        return 0

    print(f"  Loaded: {img.shape[1]}×{img.shape[0]}px")

    if deskew:
        img = deskew_sheet(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w     = img_gray.shape

    chars   = list(NEWA_CHARACTERS.items())
    n_chars = len(chars)
    print(f"  Characters: {n_chars}  Grid: {grid_cols} cols")

    all_cells = get_cells(img_gray, n_chars, write_boxes, grid_cols)

    # ── Preview mode ─────────────────────────────────────────────
    if preview_only:
        save_geometry_preview(img_gray, all_cells, image_path,
                              chars, grid_cols)
        return 0

    # ── Full extraction ──────────────────────────────────────────
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    os.makedirs(output_dir, exist_ok=True)
    total_saved = 0
    total_blank = 0
    total_aug   = 0

    for i, (class_name, _) in enumerate(chars):
        if i >= len(all_cells):
            break

        cells    = all_cells[i]
        existing = count_existing(output_dir, class_name)
        saved_n  = 0

        for x, y, cw, ch in cells:
            crop = extract_crop(img_gray, x, y, cw, ch)

            if crop is None:
                if debug:
                    cv2.rectangle(debug_img,
                                  (x,y),(x+cw,y+ch),(0,0,180),1)
                continue

            if is_blank(crop):
                total_blank += 1
                if debug:
                    cv2.rectangle(debug_img,
                                  (x,y),(x+cw,y+ch),(0,200,200),1)
                continue

            save_crop(crop, output_dir, class_name,
                      writer_id, style, "", existing, saved_n)
            saved_n     += 1
            total_saved += 1

            if debug:
                cv2.rectangle(debug_img,
                              (x,y),(x+cw,y+ch),(0,200,0),1)
                cv2.putText(debug_img, class_name[:5],
                            (x+1, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.22, (200,80,0), 1)

            if augment:
                for suffix, aug_img in augment_crop(crop):
                    save_crop(aug_img, output_dir, class_name,
                              writer_id, style, suffix,
                              existing, saved_n + total_aug)
                    total_aug += 1

    if debug:
        stem  = Path(image_path).stem
        dpath = os.path.join(DEBUG_DIR, f"{stem}_debug.jpg")
        dh, dw = debug_img.shape[:2]
        if dw > 1800:
            scale = 1800 / dw
            debug_img = cv2.resize(debug_img, (1800, int(dh*scale)))
        cv2.imwrite(dpath, debug_img)
        print(f"\n  Debug: {dpath}")
        print(f"  GREEN  = saved correctly")
        print(f"  YELLOW = blank (writer left empty)")
        print(f"  RED    = bad region")

    print(f"\n  Saved: {total_saved}  Aug: {total_aug}  Blank: {total_blank}")
    return total_saved


# ═══════════════════════════════════════════════════════════════
# MANUSCRIPT PROCESSING  (unchanged from previous version)
# ═══════════════════════════════════════════════════════════════

def process_manuscript_page(
    image_path  : str,
    output_dir  : str  = "dataset_raw/manuscript_crops",
    min_area    : int  = 400,
    max_frac    : float = 0.05,
    debug       : bool = True,
):
    print(f"\n{'='*60}")
    print(f"Manuscript: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ✗ Cannot read: {image_path}")
        return 0

    h, w      = img.shape
    page_area = h * w

    blurred  = cv2.GaussianBlur(img, (3, 3), 0)
    binary   = cv2.adaptiveThreshold(blurred, 255,
                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                   cv2.THRESH_BINARY_INV, 25, 15)

    hkernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (w//8, 1))
    hlines_m = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hkernel)
    cleaned  = cv2.subtract(binary, hlines_m)

    k_size   = max(3, min(8, w // 200))
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    dilated  = cv2.dilate(cleaned, dilate_k, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area   = cw * ch
        aspect = cw / ch if ch > 0 else 0
        if (area >= min_area and area <= page_area * max_frac
                and 0.15 < aspect < 5.0
                and cw > 15 and ch > 15):
            candidates.append((x, y, cw, ch))

    candidates.sort(key=lambda c: (c[1] // (h // 20), c[0]))
    print(f"  Found {len(candidates)} candidate regions")

    stem    = Path(image_path).stem
    out_dir = os.path.join(output_dir, "unlabeled")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for k, (x, y, cw, ch) in enumerate(candidates):
        pad  = 10
        crop = img[max(0,y-pad):y+ch+pad, max(0,x-pad):x+cw+pad]
        if crop.size == 0:
            continue
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)
        if np.median(crop) < 128:
            crop = cv2.bitwise_not(crop)
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_crop_{k:04d}.png"),
                    resized)
        saved += 1

    print(f"  Saved {saved} unlabeled crops → {out_dir}")

    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for k, (x,y,cw,ch) in enumerate(candidates):
            cv2.rectangle(debug_img,(x,y),(x+cw,y+ch),(0,180,0),1)
            cv2.putText(debug_img, str(k),(x,max(0,y-3)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,(0,0,200),1)
        dpath = os.path.join(DEBUG_DIR, f"{stem}_ms_debug.jpg")
        cv2.imwrite(dpath, debug_img)
        print(f"  Debug: {dpath}")

    return saved


def label_manuscript_crops(
    unlabeled_dir : str = "dataset_raw/manuscript_crops/unlabeled",
    output_dir    : str = "dataset_raw/manuscript_crops",
):
    import shutil
    files = sorted([f for f in os.listdir(unlabeled_dir)
                    if f.endswith('.png')])
    if not files:
        print("No unlabeled crops in:", unlabeled_dir); return

    valid  = set(NEWA_CHARACTERS.keys())
    labeled = skipped = 0
    print(f"Labeling {len(files)} crops. Type class name, SKIP, or Q.\n")

    for fname in files:
        fpath = os.path.join(unlabeled_dir, fname)
        img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        cv2.imshow(f"Label: {fname}",
                   cv2.resize(img, (256,256),
                              interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(100)
        while True:
            lbl = input(f"  {fname} → ").strip()
            if lbl.upper() == 'Q':
                cv2.destroyAllWindows()
                print(f"\nDone. Labeled: {labeled} Skipped: {skipped}")
                return
            if lbl.upper() == 'SKIP':
                skipped += 1; break
            if lbl in valid:
                out = os.path.join(output_dir, lbl)
                os.makedirs(out, exist_ok=True)
                n   = len(os.listdir(out))
                shutil.copy(fpath, os.path.join(out, f"ms_{n:04d}.png"))
                labeled += 1; break
            close = [c for c in valid if c.startswith(lbl[:2])]
            print(f"  Not found.{(' Try: '+', '.join(close[:3])) if close else ''}")
        cv2.destroyAllWindows()

    print(f"\nDone. Labeled: {labeled} Skipped: {skipped}")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary():
    sources = {
        "Handwritten Noto"    : "dataset_raw/handwritten_noto",
        "Handwritten Ranjana" : "dataset_raw/handwritten_ranjana",
        "Manuscript crops"    : "dataset_raw/manuscript_crops",
    }
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    grand = 0
    for label, path in sources.items():
        if not os.path.exists(path):
            print(f"  {label:<25}: (not created yet)")
            continue
        classes = [c for c in os.listdir(path)
                   if os.path.isdir(os.path.join(path,c))
                   and c != "unlabeled"]
        if not classes:
            print(f"  {label:<25}: (empty)"); continue
        total = sum(len([f for f in os.listdir(os.path.join(path,c))
                         if f.endswith('.png')]) for c in classes)
        avg   = total // len(classes)
        grand += total
        print(f"  {label:<25}: {total:5d} images "
              f"({len(classes)} classes, avg {avg}/class)")
    print(f"  {'TOTAL':<25}: {grand}")
    print("="*60)


# ═══════════════════════════════════════════════════════════════
# ▶  MAIN — Edit here to point to your photos
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── STEP 1: Run with PREVIEW_ONLY=True first ─────────────────
    # This saves a preview image showing where cells will be detected.
    # Open debug_crops/<name>_geometry_preview.jpg and verify that:
    #   • Green boxes sit on top of your written characters
    #   • Blue box sits on the reference Newa character (left of each group)
    # If alignment is wrong, adjust GRID_COLS or ref_frac before proceeding.

    PREVIEW_ONLY = False   # ← set True first, then False when verified

    process_sheet(
        image_path   = "handwritten_dataset/WhatsApp Image 2026-05-10 at 22.52.34.jpeg",
        output_dir   = "dataset_raw/handwritten_noto",
        writer_id    = "rajju",
        style        = "noto",
        deskew       = True,
        augment      = False,
        debug        = True,
        preview_only = PREVIEW_ONLY,
        grid_cols    = GRID_COLS,       # = 8
        write_boxes  = WRITE_BOXES,     # = 5
    )

    # Add more writers:
    # process_sheet("photos/writer2.jpg", "dataset_raw/handwritten_noto",
    #               "writer2", "noto", preview_only=PREVIEW_ONLY)

    # Manuscript processing:
    # process_manuscript_page("manuscripts/page_001.jpg", debug=True)
    # label_manuscript_crops()

    print_summary()

    print("""
IF GREEN BOXES STILL DON'T ALIGN:
══════════════════════════════════════════════════════════════
Run with PREVIEW_ONLY = True first (no crops saved).
Open debug_crops/<name>_geometry_preview.jpg.

Then tune in extract_cells_by_geometry():

  Problem: boxes shifted RIGHT (cutting into next group)
  Fix:     INCREASE ref_w fraction: 0.30 → 0.33 or 0.35

  Problem: boxes shifted LEFT (overlapping ref character)
  Fix:     DECREASE ref_w fraction: 0.30 → 0.27 or 0.25

  Problem: boxes too HIGH (in label text area)
  Fix:     DECREASE box_h fraction: 0.60 → 0.55

  Problem: wrong number of columns detected
  Fix:     Change GRID_COLS = 8 at the top to match your sheet

  Problem: first row missed entirely
  Fix:     Decrease title_frac: 0.05 → 0.03
══════════════════════════════════════════════════════════════
""")