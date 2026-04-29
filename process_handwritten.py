"""
step5a_process_sheet.py  —  Newa Script OCR
Process filled handwriting collection sheets into individual crops.

Your sheet format (from create_character_sheet.py):
  - Each row has: [shaded ref box] + [5 empty writing boxes]
  - Sheet is photographed with a phone/camera
  - This script finds the grid, crops each writing box,
    normalizes it, and saves to dataset_raw/handwritten_*/

Usage:
  python step5a_process_sheet.py

Edit the JOBS list at the bottom to point to your photos.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from newa_classes import NEWA_CHARACTERS

# ── Constants ─────────────────────────────────────────────────────
IMG_SIZE    = 128       # output image size
WRITE_BOXES = 5         # writing boxes per character on sheet
DEBUG_DIR   = "debug_crops"   # folder for debug images


# ─────────────────────────────────────────────────────────────────
# STEP 1: Deskew / perspective-correct the photo
# ─────────────────────────────────────────────────────────────────

def deskew_sheet(img):
    """
    Finds the sheet boundary in a photo and applies a
    perspective transform so the sheet fills the frame.
    Works well when the sheet is on a contrasting background.
    Returns the corrected image, or original if sheet not found.
    """
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 50, 150)

    # Dilate edges to close gaps
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    # Find largest quadrilateral contour — likely the sheet
    best = None
    best_area = 0
    img_area  = img.shape[0] * img.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * 0.1:   # must be at least 10% of image
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best      = approx
            best_area = area

    if best is None:
        print("  ⚠ Could not find sheet boundary — using full image")
        return img

    # Order corners: top-left, top-right, bottom-right, bottom-left
    pts  = best.reshape(4, 2).astype(np.float32)
    rect = order_corners(pts)

    # Target size: A4-ish proportions
    W, H = 2480, 3508   # A4 at 300 DPI
    dst  = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    corrected = cv2.warpPerspective(img, M, (W, H))
    print("  ✓ Perspective correction applied")
    return corrected


def order_corners(pts):
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]     # bottom-right: largest sum
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest diff
    return rect


# ─────────────────────────────────────────────────────────────────
# STEP 2: Detect grid lines to locate every cell
# ─────────────────────────────────────────────────────────────────

def detect_grid(img_gray, expected_cols=6, expected_rows=None,
                debug=False):
    """
    Detects the cell grid in the sheet image.
    Your sheet has 6 columns per group (1 ref + 5 write),
    and GRID_COLS=5 groups per row → 30 columns total,
    but each 'group' is visually separated.

    Returns list of (x, y, w, h) for each writing cell,
    in left-to-right, top-to-bottom order.
    """
    # Adaptive threshold to find dark lines
    thresh = cv2.adaptiveThreshold(
        img_gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10)

    # Detect horizontal and vertical lines separately
    h, w = img_gray.shape

    # Horizontal lines: long, thin
    hkernel  = cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (w // 20, 1))
    hlines   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, hkernel)

    # Vertical lines: tall, thin
    vkernel  = cv2.getStructuringElement(cv2.MORPH_RECT,
                                          (1, h // 20))
    vlines   = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vkernel)

    # Combine
    grid_mask = cv2.add(hlines, vlines)

    # Find all rectangular contours that look like cells
    contours, _ = cv2.findContours(
        grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0
        # Writing boxes are roughly square (0.6–1.6 aspect ratio)
        # and a reasonable size
        if (0.4 < aspect < 2.5
                and cw > 30 and ch > 30
                and cw < w * 0.2
                and ch < h * 0.15):
            cells.append((x, y, cw, ch))

    if debug:
        print(f"  Found {len(cells)} candidate cells")

    return sorted(cells, key=lambda c: (c[1] // 40, c[0]))


# ─────────────────────────────────────────────────────────────────
# STEP 3: Fallback — use known sheet geometry directly
# ─────────────────────────────────────────────────────────────────

def extract_cells_by_geometry(img_gray, sheet_w, sheet_h,
                               n_chars,
                               grid_cols=5,
                               ref_frac=0.083,    # ref box ~8.3% of group width
                               write_boxes=5,
                               title_frac=0.06):  # title ~6% of height
    """
    Uses the known geometry of create_character_sheet.py to
    directly compute crop coordinates.

    This is more reliable than trying to detect lines in a photo
    because photos have perspective distortion, shadows etc.

    ref_frac    : ref box width as fraction of column width
    write_boxes : number of writing boxes per character
    title_frac  : fraction of height used by title row

    Returns list of lists: cells[char_idx] = [(x,y,w,h), ...]
                           one (x,y,w,h) per writing box
    """
    h, w = img_gray.shape
    rows = (n_chars + grid_cols - 1) // grid_cols

    # Margins (estimated from sheet layout)
    # Your create_character_sheet.py uses PADDING=20 on a sheet
    # sized to fit content exactly. After perspective correction
    # to A4, estimate margins as ~2% each side.
    margin_x = int(w * 0.02)
    margin_y = int(h * 0.02)
    title_h  = int(h * title_frac)

    content_w = w - 2 * margin_x
    content_h = h - 2 * margin_y - title_h

    col_w = content_w // grid_cols
    row_h = content_h // rows

    # Within each column: ref box takes ~ref_frac, rest is writing boxes
    ref_w   = int(col_w * 0.22)
    box_gap = 4
    avail   = col_w - ref_w - 10   # 10px margin between ref and boxes
    box_w   = (avail - box_gap * (write_boxes - 1)) // write_boxes
    box_h   = int(row_h * 0.65)

    all_cells = []   # list of lists

    for i in range(n_chars):
        col = i % grid_cols
        row = i // grid_cols

        base_x = margin_x + col * col_w
        base_y = margin_y + title_h + row * row_h

        char_cells = []
        for b in range(write_boxes):
            bx = base_x + ref_w + 10 + b * (box_w + box_gap)
            by = base_y + (row_h - box_h) // 2
            char_cells.append((bx, by, box_w, box_h))

        all_cells.append(char_cells)

    return all_cells


# ─────────────────────────────────────────────────────────────────
# STEP 4: Crop, clean, and save individual character images
# ─────────────────────────────────────────────────────────────────

def clean_crop(img_gray, x, y, w, h, pad=8):
    """
    Extracts one cell, removes border lines, normalizes,
    and resizes to IMG_SIZE x IMG_SIZE.
    """
    # Add padding, clamped to image bounds
    x0 = max(0, x + pad)
    y0 = max(0, y + pad)
    x1 = min(img_gray.shape[1], x + w - pad)
    y1 = min(img_gray.shape[0], y + h - pad)

    crop = img_gray[y0:y1, x0:x1]

    if crop.size == 0:
        return None

    # Normalize contrast so ink is dark on white
    crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)

    # Make sure background is white (ink is dark)
    # If median is dark, invert
    if np.median(crop) < 128:
        crop = cv2.bitwise_not(crop)

    # Resize
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                         interpolation=cv2.INTER_AREA)
    return resized


def is_blank_cell(img, dark_threshold=200, min_dark_pixels=15):
    """True if the cell appears empty (writer skipped it)."""
    return (img < dark_threshold).sum() < min_dark_pixels


# ─────────────────────────────────────────────────────────────────
# Main processing function
# ─────────────────────────────────────────────────────────────────

def process_sheet(
    image_path: str,
    output_dir: str,
    writer_id,
    style: str = "noto",
    deskew: bool = True,
    debug: bool = True,
    write_boxes: int = WRITE_BOXES,
    grid_cols: int = 5,
):
    """
    Process one filled handwriting sheet photo.

    Args:
        image_path : path to photo of filled sheet
        output_dir : e.g. "dataset_raw/handwritten_noto"
        writer_id  : e.g. 1, 2, "rajju", etc.
        style      : "noto" or "ranjana"
        deskew     : try to perspective-correct the photo
        debug      : save a debug image showing detected cells
        write_boxes: must match WRITE_BOXES in create_character_sheet.py
        grid_cols  : must match GRID_COLS in create_character_sheet.py
    """
    print(f"\nProcessing: {image_path}")
    print(f"  Writer: {writer_id}  Style: {style}")

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ✗ Cannot read image: {image_path}")
        return 0

    # Perspective correction
    if deskew:
        img = deskew_sheet(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # Get character list in same order as the sheet was generated
    chars = list(NEWA_CHARACTERS.items())
    n_chars = len(chars)

    # Compute cell positions from sheet geometry
    all_cells = extract_cells_by_geometry(
        img_gray, w, h, n_chars,
        grid_cols=grid_cols,
        write_boxes=write_boxes)

    # Debug image
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    # Save crops
    os.makedirs(output_dir, exist_ok=True)
    total_saved  = 0
    total_blank  = 0

    for i, (class_name, _) in enumerate(chars):
        if i >= len(all_cells):
            break

        cells = all_cells[i]
        out_cls = os.path.join(output_dir, class_name)
        os.makedirs(out_cls, exist_ok=True)

        # Count existing files to avoid overwriting
        existing = len([f for f in os.listdir(out_cls)
                        if f.endswith('.png')])
        saved = 0

        for b, (x, y, cw, ch) in enumerate(cells):
            crop = clean_crop(img_gray, x, y, cw, ch)

            if crop is None:
                continue

            if is_blank_cell(crop):
                total_blank += 1
                if debug:
                    cv2.rectangle(debug_img,
                                  (x, y), (x+cw, y+ch),
                                  (200, 200, 0), 1)
                continue

            fname = (f"hw_{style}_w{writer_id}_"
                     f"{existing + saved:04d}.png")
            cv2.imwrite(os.path.join(out_cls, fname), crop)
            saved       += 1
            total_saved += 1

            if debug:
                cv2.rectangle(debug_img,
                              (x, y), (x+cw, y+ch),
                              (0, 200, 0), 1)
                cv2.putText(debug_img, class_name[:4],
                            (x+2, y+12),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, (0, 100, 200), 1)

    # Save debug image
    if debug:
        stem  = Path(image_path).stem
        dpath = os.path.join(DEBUG_DIR, f"{stem}_debug.jpg")
        cv2.imwrite(dpath, debug_img)
        print(f"  Debug image: {dpath}")
        print(f"  Green boxes = saved crops")
        print(f"  Yellow boxes = blank (skipped)")

    print(f"  ✓ Saved: {total_saved}  Blank: {total_blank}")
    return total_saved


# ─────────────────────────────────────────────────────────────────
# Interactive fallback — click to crop (for tricky photos)
# ─────────────────────────────────────────────────────────────────

def interactive_crop_session(image_path, class_name,
                              output_dir, writer_id,
                              style="noto"):
    """
    Opens an interactive window. Click top-left then bottom-right
    of each character. Good for manuscript images.

    Controls:
      Click ×2 → define crop region
      S        → save crop
      R        → reset current selection
      Q        → quit session
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return 0

    # Scale for display
    max_w, max_h = 1400, 900
    orig_h, orig_w = img.shape[:2]
    scale = min(max_w / orig_w, max_h / orig_h, 1.0)
    disp  = cv2.resize(img, (int(orig_w * scale),
                             int(orig_h * scale)))

    canvas = disp.copy()
    clicks = []
    saved  = 0

    out_cls = os.path.join(output_dir, class_name)
    os.makedirs(out_cls, exist_ok=True)
    existing = len([f for f in os.listdir(out_cls)
                    if f.endswith('.png')])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def mouse_cb(event, x, y, flags, param):
        nonlocal canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)
            if len(clicks) == 2:
                p1, p2 = clicks
                cv2.rectangle(canvas,
                              (min(p1[0], p2[0]),
                               min(p1[1], p2[1])),
                              (max(p1[0], p2[0]),
                               max(p1[1], p2[1])),
                              (0, 0, 255), 2)
            cv2.imshow("Interactive Crop", canvas)

    cv2.namedWindow("Interactive Crop")
    cv2.setMouseCallback("Interactive Crop", mouse_cb)

    print(f"\nInteractive crop — class: {class_name}")
    print("  Click TOP-LEFT then BOTTOM-RIGHT of a character")
    print("  S = save  |  R = reset  |  Q = quit")

    while True:
        cv2.imshow("Interactive Crop", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(clicks) == 2:
            # Convert display coords back to original
            p1 = (int(clicks[0][0] / scale),
                  int(clicks[0][1] / scale))
            p2 = (int(clicks[1][0] / scale),
                  int(clicks[1][1] / scale))

            x0, y0 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x1, y1 = max(p1[0], p2[0]), max(p1[1], p2[1])

            crop = gray[y0:y1, x0:x1]
            if crop.size > 0:
                crop = cv2.normalize(
                    crop, None, 0, 255, cv2.NORM_MINMAX)
                resized = cv2.resize(crop,
                                     (IMG_SIZE, IMG_SIZE))
                fname = (f"hw_{style}_w{writer_id}_"
                         f"{existing + saved:04d}.png")
                cv2.imwrite(os.path.join(out_cls, fname), resized)
                saved += 1
                print(f"  Saved #{saved}: {fname}")

            clicks.clear()
            canvas = disp.copy()

        elif key == ord('r'):
            clicks.clear()
            canvas = disp.copy()

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Session done. Saved {saved} crops for '{class_name}'")
    return saved


# ─────────────────────────────────────────────────────────────────
# Manuscript page processor
# ─────────────────────────────────────────────────────────────────

def process_manuscript_page(
    image_path: str,
    output_dir: str = "dataset_raw/manuscript_crops",
    min_char_area: int = 400,
    max_char_frac: float = 0.05,
    debug: bool = True,
):
    """
    Extracts candidate character regions from a manuscript page
    using connected component analysis.

    The output is UNLABELED crops that need manual verification.
    They are saved to:
        output_dir/unlabeled/page_<N>_crop_<M>.png

    You then manually sort them into class folders, or use
    interactive_crop_session() with a Newa reader to label them.

    Args:
        image_path    : high-res scan of manuscript page
        output_dir    : base output directory
        min_char_area : minimum pixel area to count as a character
        max_char_frac : maximum fraction of page area
        debug         : save annotated debug image
    """
    print(f"\nProcessing manuscript page: {image_path}")

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ✗ Cannot read: {image_path}")
        return 0

    h, w = img.shape
    page_area = h * w

    # Binarize
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    binary  = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 15)

    # Remove very thin lines (ruling lines on manuscript paper)
    hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    hlines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hkernel)
    cleaned = cv2.subtract(binary, hlines)

    # Connect strokes within a character
    dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    dilated  = cv2.dilate(cleaned, dilate_k, iterations=1)

    # Find connected components
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area   = cw * ch
        aspect = cw / ch if ch > 0 else 0
        if (area >= min_char_area
                and area <= page_area * max_char_frac
                and 0.15 < aspect < 5.0
                and cw > 20 and ch > 20):
            candidates.append((x, y, cw, ch))

    # Sort reading order (approximate: top-to-bottom, left-to-right)
    candidates = sorted(candidates,
                        key=lambda c: (c[1] // 60, c[0]))

    # Save unlabeled crops
    stem    = Path(image_path).stem
    out_dir = os.path.join(output_dir, "unlabeled")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for k, (x, y, cw, ch) in enumerate(candidates):
        pad  = 12
        crop = img[max(0, y-pad): y+ch+pad,
                   max(0, x-pad): x+cw+pad]
        if crop.size == 0:
            continue

        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX)

        # Make sure ink is dark
        if np.median(crop) < 128:
            crop = cv2.bitwise_not(crop)

        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_AREA)
        fname   = f"{stem}_crop_{k:04d}.png"
        cv2.imwrite(os.path.join(out_dir, fname), resized)
        saved += 1

    print(f"  ✓ Extracted {saved} candidate character regions")
    print(f"  Saved to: {out_dir}")
    print(f"  NEXT: Sort these into class folders manually,")
    print(f"        or use interactive_crop_session() with a reader.")

    # Debug image
    if debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for k, (x, y, cw, ch) in enumerate(candidates):
            cv2.rectangle(debug_img,
                          (x, y), (x+cw, y+ch),
                          (0, 180, 0), 1)
            cv2.putText(debug_img, str(k),
                        (x, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 200), 1)
        dpath = os.path.join(DEBUG_DIR, f"{stem}_manuscript_debug.jpg")
        cv2.imwrite(dpath, debug_img)
        print(f"  Debug image (all detected regions): {dpath}")

    return saved


# ─────────────────────────────────────────────────────────────────
# Manuscript labeling helper
# ─────────────────────────────────────────────────────────────────

def label_manuscript_crops(
    unlabeled_dir: str = "dataset_raw/manuscript_crops/unlabeled",
    output_dir: str    = "dataset_raw/manuscript_crops",
):
    """
    Interactive labeling tool for manuscript crops.

    Shows each unlabeled crop, you type the class name to save it,
    or press SKIP to skip, or Q to quit.

    Run this with a Newa reader sitting next to you.
    """
    files = sorted([
        f for f in os.listdir(unlabeled_dir)
        if f.endswith('.png')
    ])

    if not files:
        print("No unlabeled crops found.")
        return

    valid_classes = set(NEWA_CHARACTERS.keys())
    print(f"\nLabeling {len(files)} manuscript crops")
    print("Type class name (e.g. 'ka', 'matra_i') and press Enter")
    print("Type SKIP to skip, Q to quit\n")

    labeled = 0
    skipped = 0

    for fname in files:
        fpath = os.path.join(unlabeled_dir, fname)
        img   = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Show the crop (scaled up for visibility)
        display = cv2.resize(img, (256, 256),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Label: {fname}", display)
        cv2.waitKey(100)

        # Get label from user
        while True:
            label = input(f"  {fname} → class: ").strip()

            if label.upper() == 'Q':
                cv2.destroyAllWindows()
                print(f"\nStopped. Labeled: {labeled}  Skipped: {skipped}")
                return

            if label.upper() == 'SKIP':
                skipped += 1
                break

            if label in valid_classes:
                out_cls = os.path.join(output_dir, label)
                os.makedirs(out_cls, exist_ok=True)
                existing = len(os.listdir(out_cls))
                dst = os.path.join(out_cls, f"ms_{existing:04d}.png")
                import shutil
                shutil.copy(fpath, dst)
                labeled += 1
                print(f"    Saved as {dst}")
                break
            else:
                print(f"    '{label}' not in NEWA_CHARACTERS. "
                      f"Try again or type SKIP.")

        cv2.destroyAllWindows()

    print(f"\nDone. Labeled: {labeled}  Skipped: {skipped}")


# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────

def print_summary():
    sources = {
        "Handwritten Noto":     "dataset_raw/handwritten_noto",
        "Handwritten Ranjana":  "dataset_raw/handwritten_ranjana",
        "Manuscript crops":     "dataset_raw/manuscript_crops",
    }
    print("\n" + "="*60)
    print("DATA COLLECTION SUMMARY")
    print("="*60)
    for label, path in sources.items():
        if not os.path.exists(path):
            print(f"  {label:<25}: not yet created")
            continue
        classes = [c for c in os.listdir(path)
                   if os.path.isdir(os.path.join(path, c))
                   and c != "unlabeled"]
        total   = sum(
            len(os.listdir(os.path.join(path, c)))
            for c in classes)
        avg     = total // len(classes) if classes else 0
        print(f"  {label:<25}: {total:5} images  "
              f"({len(classes)} classes, avg {avg}/class)")
    print("="*60)


# ─────────────────────────────────────────────────────────────────
# MAIN — Edit this section to process your photos
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5A — SHEET + MANUSCRIPT PROCESSING")
    print("=" * 60)

    # ── HANDWRITING SHEET PROCESSING ─────────────────────────────
    # Edit these paths to match your actual photo files.
    # writer_id can be a number or a name string.

    # Example: Process Rajju's sheet
    process_sheet(
        image_path = "handwritten_dataset/Screenshot 2026-04-29 180702.png",
        output_dir = "dataset_raw/handwritten_noto",
        writer_id  = "rajju",
        style      = "noto",
        deskew     = True,
        debug      = True,
    )

    # Example: Process multiple writers in a loop
    # writers = [
    #     ("photos/writer1_sheet.jpg", "w1", "noto"),
    #     ("photos/writer2_sheet.jpg", "w2", "ranjana"),
    #     ("photos/writer3_sheet.jpg", "w3", "ranjana"),
    # ]
    # for path, wid, style in writers:
    #     process_sheet(path,
    #                   f"dataset_raw/handwritten_{style}",
    #                   wid, style,
    #                   deskew=True, debug=True)

    # ── MANUSCRIPT PROCESSING ─────────────────────────────────────
    # Step 1: Extract candidate regions from a manuscript page
    # process_manuscript_page(
    #     image_path = "manuscripts/page_001.jpg",
    #     output_dir = "dataset_raw/manuscript_crops",
    #     debug      = True,
    # )

    # Step 2: Label them (sit with a Newa reader)
    # label_manuscript_crops(
    #     unlabeled_dir = "dataset_raw/manuscript_crops/unlabeled",
    #     output_dir    = "dataset_raw/manuscript_crops",
    # )

    # ── SHOW CURRENT SUMMARY ─────────────────────────────────────
    print_summary()

    print("""
HOW TO USE THIS SCRIPT
══════════════════════════════════════════════════════════════

FOR HANDWRITING SHEETS (like Rajju's):
  1. Put your sheet photos in a  photos/  folder
  2. Uncomment the process_sheet() call above
  3. Set image_path, writer_id, style
  4. Run:  python step5a_process_sheet.py
  5. Check debug_crops/ folder — you'll see green boxes
     around detected writing cells. If they're wrong, see
     TROUBLESHOOTING below.

FOR MANUSCRIPT PAGES:
  1. Get a scan (high-res JPG or TIFF) from an archive
  2. Uncomment process_manuscript_page() with your scan path
  3. Run the script — it saves unlabeled crops
  4. Sit with a Newa reader and run label_manuscript_crops()
     to assign class names to each crop

TROUBLESHOOTING:
  If the green boxes don't line up with your writing cells:
  → Open the debug image in debug_crops/
  → Count actual pixels for a few boxes
  → Adjust the fractions in extract_cells_by_geometry():
      ref_frac  (default 0.22)
      box sizes are computed automatically
  → Or use interactive_crop_session() to click each one manually

OUTPUT FOLDERS:
  dataset_raw/handwritten_noto/      ← modern style writers
  dataset_raw/handwritten_ranjana/   ← Prachalit style writers
  dataset_raw/manuscript_crops/      ← historical manuscript crops
══════════════════════════════════════════════════════════════
""")