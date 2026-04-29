# step5_process_handwritten.py
# ─────────────────────────────────────────────────────────────────
# Processes photos of filled handwriting sheets.
# Extracts individual character crops and saves them.
#
# Two modes:
#   MODE A — Single class per photo (easiest)
#             Write 'ka' 25 times, photograph, process.
#   MODE B — Full sheet (one photo of entire sheet)
#             Harder to auto-process, but saves printing time.
#
# Output:
#   dataset_raw/handwritten_noto/    ← from Noto-style writers
#   dataset_raw/handwritten_ranjana/ ← from Ranjana-style writers
#
# Usage:  python step5_process_handwritten.py
# ─────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from newa_classes import NEWA_CHARACTERS

IMG_SIZE = 128

# ─────────────────────────────────────────────────────────────────
# MODE A — Single class photo processor (RECOMMENDED)
# Write one character ~25 times on paper, photograph, process.
# ─────────────────────────────────────────────────────────────────

def process_single_class_photo(
    image_path: str,
    class_name: str,
    output_dir: str,
    writer_id: int | str,
    style: str = "noto",          # "noto" or "ranjana"
    expected: int = 25,           # how many chars you wrote
    min_area: int = 300,
    debug: bool = False
):
    """
    Extracts character crops from a photo where you wrote
    the same character multiple times in a grid.

    Args:
        image_path : path to your photo
        class_name : e.g. "ka", "matra_i", "digit_3"
        output_dir : base output directory
        writer_id  : number or name identifying the writer
        style      : "noto" or "ranjana"
        expected   : how many characters you wrote
        debug      : if True, saves a debug image showing contours
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  ✗ Cannot read: {image_path}")
        return 0

    # Preprocessing
    blur    = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connect nearby strokes
    kernel  = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    # Filter by size
    img_area = img.shape[0] * img.shape[1]
    valid    = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if (area > min_area
                and area < img_area * 0.25
                and w > 15 and h > 15):
            valid.append((x, y, w, h))

    # Sort top-left to bottom-right (reading order)
    valid = sorted(valid, key=lambda b: (b[1] // 60, b[0]))

    if debug:
        debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for j, (x, y, w, h) in enumerate(valid):
            cv2.rectangle(debug_img,
                          (x,y), (x+w, y+h),
                          (0,255,0), 2)
            cv2.putText(debug_img, str(j),
                        (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,0,255), 1)
        dbg_path = image_path.replace('.', '_debug.')
        cv2.imwrite(dbg_path, debug_img)
        print(f"  Debug image: {dbg_path}")

    # Save crops
    out_cls = os.path.join(output_dir, class_name)
    os.makedirs(out_cls, exist_ok=True)

    # Count existing to avoid overwriting
    existing = len([f for f in os.listdir(out_cls)
                    if f.endswith('.png')])
    saved    = 0

    for j, (x, y, w, h) in enumerate(valid[:expected]):
        pad  = 10
        crop = img[
            max(0, y-pad): y + h + pad,
            max(0, x-pad): x + w + pad
        ]
        resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

        # Normalize contrast
        resized = cv2.normalize(
            resized, None, 0, 255, cv2.NORM_MINMAX)

        fname    = f"hw_{style}_w{writer_id}_{existing+j:04d}.png"
        out_path = os.path.join(out_cls, fname)
        cv2.imwrite(out_path, resized)
        saved += 1

    print(f"  [{class_name}] writer={writer_id} style={style}: "
          f"extracted {saved}/{len(valid)} chars "
          f"(found {len(valid)} contours)")
    return saved


# ─────────────────────────────────────────────────────────────────
# MODE B — Interactive single-character cropper
# Run this with a photo open, click to define crops.
# ─────────────────────────────────────────────────────────────────

def interactive_crop(image_path, class_name, output_dir,
                     writer_id, style="noto"):
    """
    Opens a window. Click top-left then bottom-right of each
    character. Press S to save, R to reset, Q to quit.
    Good for manuscript images or difficult photos.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read: {image_path}")
        return

    # Resize display if too large
    max_w, max_h = 1400, 900
    h, w = img.shape[:2]
    scale = min(max_w/w, max_h/h, 1.0)
    if scale < 1.0:
        img_disp = cv2.resize(img,
                              (int(w*scale), int(h*scale)))
    else:
        img_disp = img.copy()

    canvas  = img_disp.copy()
    clicks  = []
    saved   = 0

    def mouse_cb(event, x, y, flags, param):
        nonlocal canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(canvas, (x,y), 4, (0,255,0), -1)
            if len(clicks) == 2:
                x1,y1 = clicks[0]
                x2,y2 = clicks[1]
                cv2.rectangle(canvas,
                              (min(x1,x2), min(y1,y2)),
                              (max(x1,x2), max(y1,y2)),
                              (0,0,255), 2)
            cv2.imshow("Crop Tool", canvas)

    cv2.imshow("Crop Tool", canvas)
    cv2.setMouseCallback("Crop Tool", mouse_cb)

    out_cls = os.path.join(output_dir, class_name)
    os.makedirs(out_cls, exist_ok=True)
    existing = len(os.listdir(out_cls))

    print(f"\nInteractive crop — class: {class_name}")
    print("  Click TOP-LEFT then BOTTOM-RIGHT of a character")
    print("  S = save crop  |  R = reset  |  Q = quit")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    while True:
        cv2.imshow("Crop Tool", canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(clicks) == 2:
            x1, y1 = [int(c / scale) for c in clicks[0]]
            x2, y2 = [int(c / scale) for c in clicks[1]]

            crop    = gray[min(y1,y2):max(y1,y2),
                           min(x1,x2):max(x1,x2)]
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            norm    = cv2.normalize(
                resized, None, 0, 255, cv2.NORM_MINMAX)

            fname   = (f"hw_{style}_w{writer_id}_"
                       f"{existing+saved:04d}.png")
            cv2.imwrite(os.path.join(out_cls, fname), norm)
            saved  += 1
            print(f"  Saved #{saved}: {fname}")

            clicks.clear()
            canvas = img_disp.copy()

        elif key == ord('r'):
            clicks.clear()
            canvas = img_disp.copy()

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Session done. Saved {saved} crops for '{class_name}'")
    return saved


# ─────────────────────────────────────────────────────────────────
# Batch processor — process a whole folder of photos
# ─────────────────────────────────────────────────────────────────

def batch_process_folder(
    photos_dir: str,
    output_dir: str,
    writer_id: int | str,
    style: str,
    expected: int = 25
):
    """
    Process all photos in a folder.
    File naming convention: ka_01.jpg, matra_i_02.jpg, etc.
    The class name is taken from the filename (before last underscore).

    Example folder:
        handwriting_photos/
            ka_01.jpg        → class: ka
            kha_01.jpg       → class: kha
            matra_i_01.jpg   → class: matra_i
    """
    if not os.path.exists(photos_dir):
        print(f"Photos folder not found: {photos_dir}")
        return

    photos = [f for f in os.listdir(photos_dir)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    if not photos:
        print(f"No photos found in {photos_dir}")
        return

    print(f"\nBatch processing {len(photos)} photos...")
    total = 0

    for fname in sorted(photos):
        # Extract class name from filename
        # e.g. "ka_01.jpg" → "ka"
        stem       = os.path.splitext(fname)[0]
        parts      = stem.rsplit('_', 1)
        class_name = parts[0] if len(parts) > 1 else stem

        if class_name not in NEWA_CHARACTERS:
            print(f"  ⚠ Skipping {fname} — '{class_name}' "
                  f"not in NEWA_CHARACTERS")
            continue

        img_path = os.path.join(photos_dir, fname)
        n = process_single_class_photo(
            img_path, class_name, output_dir,
            writer_id, style, expected)
        total += n

    print(f"\nBatch complete. Total crops: {total}")
    return total


# ─────────────────────────────────────────────────────────────────
# Dataset summary
# ─────────────────────────────────────────────────────────────────

def print_handwritten_summary():
    dirs = [
        "dataset_raw/handwritten_noto",
        "dataset_raw/handwritten_ranjana",
    ]
    print("\nHandwritten data summary:")
    for d in dirs:
        if not os.path.exists(d):
            print(f"  {d}: not yet created")
            continue
        classes  = [c for c in os.listdir(d)
                    if os.path.isdir(os.path.join(d, c))]
        total    = sum(
            len(os.listdir(os.path.join(d, c)))
            for c in classes)
        avg      = total // len(classes) if classes else 0
        print(f"  {d}:")
        print(f"    Classes: {len(classes)}")
        print(f"    Total  : {total}")
        print(f"    Avg/cls: {avg}")


# ─────────────────────────────────────────────────────────────────
# MAIN — Example usage
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 5 — HANDWRITTEN DATA PROCESSING")
    print("=" * 60)

    print("""
─────────────────────────────────────────
HOW TO USE THIS SCRIPT
─────────────────────────────────────────

OPTION 1 — Single class photo (EASIEST):
  Write 'ka' 25 times on paper
  Photograph it
  Edit this script and add:

    process_single_class_photo(
        image_path = "photos/ka_writer1.jpg",
        class_name = "ka",
        output_dir = "dataset_raw/handwritten_noto",
        writer_id  = 1,
        style      = "noto",
        expected   = 25,
        debug      = True   # shows detection boxes
    )

OPTION 2 — Batch process a folder of photos:
  Name files like:  ka_01.jpg, kha_01.jpg, ga_01.jpg
  Then run:

    batch_process_folder(
        photos_dir = "photos/writer1_noto/",
        output_dir = "dataset_raw/handwritten_noto",
        writer_id  = 1,
        style      = "noto",
        expected   = 25,
    )

OPTION 3 — Interactive crop (for difficult photos):
  Opens a window, you click each character manually:

    interactive_crop(
        image_path = "photos/manuscript.jpg",
        class_name = "ka",
        output_dir = "dataset_raw/handwritten_ranjana",
        writer_id  = "ranjana_w1",
        style      = "ranjana",
    )

─────────────────────────────────────────
HOW MANY WRITERS DO YOU NEED?
─────────────────────────────────────────
  Noto style   : 7 writers already done ✓
                 Target: 7 writers × 82 chars × 5 samples
                       = 2,870 handwritten images

  Ranjana style: 5 writers planned
                 Target: 5 writers × 82 chars × 5 samples
                       = 2,050 handwritten images

  Total target : ~4,920 handwritten images
  This is very solid for a student thesis.

─────────────────────────────────────────
""")

    print_handwritten_summary()

    # ── Uncomment and edit to process your photos ─────────────────
    # Example: process one photo
    # process_single_class_photo(
    #     image_path = "photos/ka_writer1.jpg",
    #     class_name = "ka",
    #     output_dir = "dataset_raw/handwritten_noto",
    #     writer_id  = 1,
    #     style      = "noto",
    #     expected   = 25,
    #     debug      = True,
    # )

    # Example: batch process a writer's folder
    # batch_process_folder(
    #     photos_dir = "photos/writer8_ranjana/",
    #     output_dir = "dataset_raw/handwritten_ranjana",
    #     writer_id  = 8,
    #     style      = "ranjana",
    #     expected   = 25,
    # )