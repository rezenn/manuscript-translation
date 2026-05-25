"""
segment.py  —  Manuscript Page Segmentation
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
Takes a photo of a Newa manuscript page and breaks it down into
individual character images that your OCR model can recognize.

THE PROBLEM
───────────
A manuscript photo is one big image. Your OCR model expects
small 64×64 images of individual characters. We need to:
  1. Fix the tilt/rotation of the page (deskew)
  2. Clean up the image (binarize — make it black and white)
  3. Find where the text lines are
  4. Within each line, find individual character bounding boxes
  5. Crop and save each character as a small image

HOW DESKEWING WORKS
───────────────────
Old manuscripts are often photographed at a slight angle.
We use the Hough line transform to detect the dominant angle
of ink strokes, then rotate the image to make them horizontal.

HOW CHARACTER DETECTION WORKS
──────────────────────────────
After binarizing the image (black ink on white background):
  1. We project pixel darkness onto the vertical axis
     → This creates a "row profile" — tall spikes = text lines
  2. We find the gaps between spikes → these are line boundaries
  3. Within each line strip, we project onto the horizontal axis
     → Spikes = character columns
  4. We group overlapping column projections into bounding boxes
  5. Each bounding box = one character crop

WHAT YOU GET
────────────
A folder like:
    output_segments/
        line_00_char_000.png
        line_00_char_001.png
        line_01_char_000.png
        ...
    segments_meta.json   ← positions of every character (for reassembly)

Run with:
    python transliteration/segment.py --image path/to/manuscript.jpg
    python transliteration/segment.py --image manuscript.jpg --output my_segments/ --debug
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# STEP 1: LOAD AND PREPROCESS
# ═══════════════════════════════════════════════════════════════════

def load_image(image_path: str):
    """
    Load image from disk and convert to grayscale.
    Grayscale = one channel (0-255) instead of three (RGB).
    We don't need colour for OCR.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  Loaded: {image_path}  ({img.shape[1]}×{img.shape[0]} px)")
    return img, gray


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Convert grayscale → pure black/white (binary) image.

    We use Otsu's method: it automatically finds the best threshold
    value that separates ink (dark) from paper (light).

    Also applies adaptive thresholding as a fallback for uneven
    lighting (common in manuscript photos — shadows near edges).
    """
    # Blur slightly to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu global threshold
    _, otsu = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Adaptive threshold (handles uneven illumination)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )

    # Combine: pixel is "ink" if EITHER method thinks so
    binary = cv2.bitwise_or(otsu, adaptive)
    return binary


# ═══════════════════════════════════════════════════════════════════
# STEP 2: DESKEW (FIX PAGE ROTATION)
# ═══════════════════════════════════════════════════════════════════

def deskew(gray: np.ndarray, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Detect and correct page tilt.

    Method: find the angle that minimises the variance of each
    row's pixel sum. When the page is perfectly horizontal,
    text rows create sharp dark bands → high variance.
    A tilted page smears those bands → low variance.

    Returns: deskewed gray, deskewed binary, angle in degrees
    """
    # Only look at strong ink pixels for angle detection
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        print("  Deskew: not enough ink pixels, skipping")
        return gray, binary, 0.0

    # minAreaRect finds the rectangle that best fits the ink blob
    # The angle of that rectangle approximates the page tilt
    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in [-90, 0]; normalise to [-45, 45]
    if angle < -45:
        angle = 90 + angle
    else:
        angle = angle  # already in range

    # Only correct if tilt is more than 0.5 degrees
    if abs(angle) < 0.5:
        print(f"  Deskew: tilt={angle:.2f}° (negligible, skipping)")
        return gray, binary, angle

    print(f"  Deskew: correcting {angle:.2f}° rotation")

    h, w = gray.shape
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)

    gray_deskewed = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    binary_deskewed = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return gray_deskewed, binary_deskewed, angle


# ═══════════════════════════════════════════════════════════════════
# STEP 3: FIND TEXT LINES
# ═══════════════════════════════════════════════════════════════════

def find_text_lines(binary: np.ndarray, min_line_height: int = 15) -> list[tuple[int, int]]:
    """
    Find the vertical extents (top, bottom) of each text line.

    Method: horizontal projection profile.
    - Sum each row of the binary image → how many ink pixels in that row
    - Text lines = rows with many ink pixels (high sum)
    - Gaps between lines = rows with few/no ink pixels (low sum)

    Returns: list of (top_row, bottom_row) for each line
    """
    # Sum ink pixels per row → 1D array
    row_sums = binary.sum(axis=1)

    # Smooth to remove noise
    kernel = np.ones(5) / 5
    row_sums_smooth = np.convolve(row_sums, kernel, mode='same')

    # Threshold: a row is "text" if it has > 1% of max ink density
    threshold = row_sums_smooth.max() * 0.01
    is_text_row = row_sums_smooth > threshold

    # Find transitions: gap→text (line start) and text→gap (line end)
    lines = []
    in_line = False
    line_start = 0

    for i, is_text in enumerate(is_text_row):
        if is_text and not in_line:
            in_line = True
            line_start = i
        elif not is_text and in_line:
            in_line = False
            line_height = i - line_start
            if line_height >= min_line_height:
                # Add a small margin above and below
                top    = max(0, line_start - 3)
                bottom = min(binary.shape[0], i + 3)
                lines.append((top, bottom))

    # Handle case where text goes to the bottom of the image
    if in_line:
        line_height = binary.shape[0] - line_start
        if line_height >= min_line_height:
            lines.append((max(0, line_start - 3), binary.shape[0]))

    print(f"  Found {len(lines)} text lines")
    return lines


# ═══════════════════════════════════════════════════════════════════
# STEP 4: FIND CHARACTERS WITHIN EACH LINE
# ═══════════════════════════════════════════════════════════════════

def find_characters_in_line(
    binary_line: np.ndarray,
    min_char_width: int = 8,
    min_char_height: int = 8,
) -> list[tuple[int, int, int, int]]:
    """
    Find individual character bounding boxes within one text line strip.

    Two methods tried in order:
    1. Connected components — each blob of connected ink pixels is a character
       (works great for clearly separated characters)
    2. Column projection — fallback for touching/overlapping characters

    Returns: list of (x, y, w, h) bounding boxes, left to right
    """
    # Method 1: connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_line, connectivity=8
    )

    boxes = []
    for i in range(1, num_labels):  # skip label 0 = background
        x, y, w, h, area = stats[i]
        if w >= min_char_width and h >= min_char_height and area >= 30:
            boxes.append((x, y, w, h))

    # If we got very few components, try column projection as fallback
    if len(boxes) < 2:
        col_sums = binary_line.sum(axis=0)
        kernel   = np.ones(3) / 3
        col_smooth = np.convolve(col_sums, kernel, mode='same')
        threshold  = col_smooth.max() * 0.05 if col_smooth.max() > 0 else 1

        in_char   = False
        char_start = 0
        boxes = []
        for i, val in enumerate(col_smooth):
            if val > threshold and not in_char:
                in_char    = True
                char_start = i
            elif val <= threshold and in_char:
                in_char = False
                width = i - char_start
                if width >= min_char_width:
                    boxes.append((char_start, 0, width, binary_line.shape[0]))

    # Sort left to right
    boxes.sort(key=lambda b: b[0])
    return boxes


# ═══════════════════════════════════════════════════════════════════
# STEP 5: MERGE OVERLAPPING BOXES (handles diacritics/vowel marks)
# ═══════════════════════════════════════════════════════════════════

def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    overlap_threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """
    Newa script has vowel marks (matras) that sit above or below the
    base character. Connected components may detect them as separate blobs.

    This function merges horizontally overlapping boxes into one,
    so each (base character + its diacritics) becomes a single box.

    overlap_threshold: if two boxes share more than this fraction of
    horizontal extent, merge them.
    """
    if not boxes:
        return boxes

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        result  = []
        used    = [False] * len(merged)

        for i, (x1, y1, w1, h1) in enumerate(merged):
            if used[i]:
                continue
            # Start a group with box i
            gx1, gy1 = x1, y1
            gx2, gy2 = x1 + w1, y1 + h1

            for j, (x2, y2, w2, h2) in enumerate(merged):
                if i == j or used[j]:
                    continue
                # Check horizontal overlap
                ox1 = max(gx1, x2)
                ox2 = min(gx2, x2 + w2)
                if ox2 - ox1 > 0:
                    overlap = (ox2 - ox1) / min(gx2 - gx1, w2)
                    if overlap >= overlap_threshold:
                        # Merge j into the current group
                        gx1 = min(gx1, x2)
                        gy1 = min(gy1, y2)
                        gx2 = max(gx2, x2 + w2)
                        gy2 = max(gy2, y2 + h2)
                        used[j] = True
                        changed  = True

            used[i] = True
            result.append((gx1, gy1, gx2 - gx1, gy2 - gy1))

        merged = result

    return merged


# ═══════════════════════════════════════════════════════════════════
# STEP 6: CROP AND SAVE CHARACTER IMAGES
# ═══════════════════════════════════════════════════════════════════

def crop_and_save(
    gray: np.ndarray,
    binary: np.ndarray,
    lines: list[tuple[int, int]],
    output_dir: str,
    padding: int = 4,
    target_size: int = 64,
) -> list[dict]:
    """
    For each character bounding box, crop the image and save it.
    Also returns metadata (position, filename) for later reassembly.

    padding: add N pixels of white space around each character crop
    target_size: resize to this size (must match what your model expects)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = []
    total_chars = 0

    for line_idx, (line_top, line_bottom) in enumerate(lines):
        # Crop the binary image to this line strip
        line_binary = binary[line_top:line_bottom, :]
        line_gray   = gray[line_top:line_bottom, :]

        # Find characters in this line
        boxes = find_characters_in_line(line_binary)
        boxes = merge_overlapping_boxes(boxes)

        for char_idx, (x, y, w, h) in enumerate(boxes):
            # Add padding (clamp to image boundaries)
            img_h, img_w = line_gray.shape
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img_w, x + w + padding)
            y2 = min(img_h, y + h + padding)

            # Crop from the grayscale image
            char_crop = line_gray[y1:y2, x1:x2]

            if char_crop.size == 0:
                continue

            # Resize to target_size × target_size (what the model expects)
            char_resized = cv2.resize(
                char_crop,
                (target_size, target_size),
                interpolation=cv2.INTER_AREA
            )

            # Save
            filename = f"line_{line_idx:02d}_char_{char_idx:03d}.png"
            save_path = out_path / filename
            cv2.imwrite(str(save_path), char_resized)

            # Record metadata
            metadata.append({
                "file":      filename,
                "line":      line_idx,
                "char_idx":  char_idx,
                "page_x":    int(x1),
                "page_y":    int(line_top + y1),
                "width":     int(x2 - x1),
                "height":    int(y2 - y1),
                "predicted": None,   # filled in by recognize.py
            })
            total_chars += 1

    print(f"  Saved {total_chars} character crops → {output_dir}/")
    return metadata


# ═══════════════════════════════════════════════════════════════════
# OPTIONAL: DEBUG VISUALISATION
# ═══════════════════════════════════════════════════════════════════

def save_debug_image(
    original: np.ndarray,
    lines: list[tuple[int, int]],
    metadata: list[dict],
    out_path: str,
):
    """
    Draw line boundaries and character boxes on the original image.
    Useful for checking that segmentation is working correctly.
    """
    debug = original.copy()

    # Draw line boundaries in blue
    for top, bottom in lines:
        cv2.line(debug, (0, top),    (debug.shape[1], top),    (255, 100, 0), 1)
        cv2.line(debug, (0, bottom), (debug.shape[1], bottom), (255, 100, 0), 1)

    # Draw character boxes in green
    for item in metadata:
        x = item["page_x"]
        y = item["page_y"]
        w = item["width"]
        h = item["height"]
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 200, 0), 1)

    cv2.imwrite(out_path, debug)
    print(f"  Debug image saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def segment_page(
    image_path: str,
    output_dir: str   = "output_segments",
    target_size: int  = 64,
    debug: bool       = False,
) -> list[dict]:
    """
    Full segmentation pipeline. Returns metadata list.
    """
    print(f"\n{'─'*55}")
    print(f"  Segmenting: {image_path}")
    print(f"{'─'*55}")

    # 1. Load
    original, gray = load_image(image_path)

    # 2. Binarize
    print("  Binarizing...")
    binary = binarize(gray)

    # 3. Deskew
    print("  Deskewing...")
    gray, binary, angle = deskew(gray, binary)

    # 4. Find lines
    print("  Finding text lines...")
    lines = find_text_lines(binary)

    if not lines:
        print("  WARNING: No text lines detected. Check image quality.")
        return []

    # 5. Crop characters
    print("  Extracting characters...")
    metadata = crop_and_save(gray, binary, lines, output_dir, target_size=target_size)

    # 6. Save metadata
    meta_path = Path(output_dir) / "segments_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_image": str(image_path),
            "deskew_angle": round(angle, 3),
            "num_lines":    len(lines),
            "num_chars":    len(metadata),
            "characters":   metadata,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved → {meta_path}")

    # 7. Optional debug image
    if debug:
        debug_path = str(Path(output_dir) / "debug_segmentation.jpg")
        # Rebuild original at deskewed angle for overlay
        save_debug_image(
            cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
            lines, metadata, debug_path
        )

    print(f"\n  ✓ Segmentation complete: {len(metadata)} characters from {len(lines)} lines")
    return metadata


def parse_args():
    p = argparse.ArgumentParser(description="Segment a Newa manuscript page into characters")
    p.add_argument("--image",   required=True, help="Path to manuscript image (jpg/png/tif)")
    p.add_argument("--output",  default="output_segments", help="Output directory for crops")
    p.add_argument("--size",    type=int, default=64, help="Output character image size (default: 64)")
    p.add_argument("--debug",   action="store_true", help="Save debug image with bounding boxes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    segment_page(args.image, args.output, args.size, args.debug)