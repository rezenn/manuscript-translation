"""
segment.py  —  Newa Manuscript Page Segmentation (v5)
══════════════════════════════════════════════════════════════════

FIXES vs v4
───────────
v5 key improvements:
  1. Stronger red/orange channel suppression using HSV red-mask.
  2. Smarter noise filtering — min_area raised, aspect-ratio filter added.
  3. Better border crop — Phase 2 threshold loosened to 60%.
  4. Diacritic merge distance increased for high-res images.
  5. find_characters_in_line filters on BOTH min size AND max aspect ratio.
  6. Panel detection gap threshold scaled to image height.
  7. Auto valley threshold clamped more aggressively.

PARAMETER NOTE
──────────────
  segment_page() uses `valley_threshold` (not seg_threshold).
  translate.py must pass `valley_threshold=args.seg_threshold`.

Run:
    python transliteration/segment.py --image manuscript.jpg --debug
    python transliteration/segment.py --image manuscript.jpg --debug --threshold 0.20
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# ══════════════════════════════════════════════════════════════════
# HELPER: RED-SUPPRESSED BINARIZATION
# ══════════════════════════════════════════════════════════════════

def make_binary(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image → ink mask (ink=255, background=0).

    Strategy:
      1. Build a red-pixel mask. These are ruling lines / red decorative
         ink — zero them out before binarization.
      2. Use B+G channel average — suppresses remaining orange/red tones.
      3. Combine Otsu + adaptive threshold for robust binarization.
      4. Morphological opening removes isolated noise pixels.
    """
    # 1. Suppress red/orange pixels via HSV mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red_mask1   = cv2.inRange(hsv, (0,   80,  60), (15,  255, 255))
    red_mask2   = cv2.inRange(hsv, (160, 80,  60), (180, 255, 255))
    orange_mask = cv2.inRange(hsv, (8,   60,  60), (25,  255, 255))
    red_mask = cv2.bitwise_or(cv2.bitwise_or(red_mask1, red_mask2), orange_mask)
    k_red = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.dilate(red_mask, k_red, iterations=1)

    clean = img.copy()
    clean[red_mask > 0] = [255, 255, 255]

    # 2. B+G average
    b = clean[:, :, 0].astype(np.float32)
    g = clean[:, :, 1].astype(np.float32)
    bg = ((b + g) / 2).astype(np.uint8)
    blurred = cv2.GaussianBlur(bg, (3, 3), 0)

    # 3. Otsu + adaptive
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=31, C=10,
    )
    binary = cv2.bitwise_or(otsu, adaptive)

    # 4. Morphological opening
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    # 5. Zero out red pixels in final output
    binary[red_mask > 0] = 0
    return binary


# ══════════════════════════════════════════════════════════════════
# STEP 1 — SMART PAGE CROP
# ══════════════════════════════════════════════════════════════════

def smart_crop(img: np.ndarray, min_upscale_height: int = 600):
    """
    Crop dark borders. Returns (cropped_img, (x_offset, y_offset)).

    Phase 1: bright-region crop — removes dark photo background.
    Phase 2: strip ornamental column borders (threshold 60%).
    Phase 3: upscale if image is too small for valley detection.
    """
    h0, w0 = img.shape[:2]

    # Phase 1: bright-region crop
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = 0, 0
    if contours:
        rx, ry, rw, rh = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if rw * rh > 0.4 * w0 * h0:
            mx = max(3, int(rw * 0.01))
            my = max(3, int(rh * 0.01))
            x1, y1 = max(0, rx + mx), max(0, ry + my)
            x2, y2 = min(w0, rx + rw - mx), min(h0, ry + rh - my)
            img = img[y1:y2, x1:x2]
            ox, oy = x1, y1
            print(f"  Phase-1 crop: ({x1},{y1})→({x2},{y2})")

    # Phase 2: strip ornamental column borders
    h, w = img.shape[:2]
    binary = make_binary(img)
    mid_binary = binary[int(h * 0.20): int(h * 0.80), :]
    col_ink = mid_binary.sum(axis=0).astype(float) / max(1, mid_binary.shape[0])
    max_col = col_ink.max()

    left_crop, right_crop, top_crop, bot_crop = 0, w, 0, h

    if max_col > 0:
        border_thresh = max_col * 0.60

        for c in range(w):
            if col_ink[c] < border_thresh:
                left_crop = c
                break
        for c in range(w - 1, -1, -1):
            if col_ink[c] < border_thresh:
                right_crop = c + 1
                break

        mid_binary2 = binary[:, left_crop:right_crop]
        row_ink = mid_binary2.sum(axis=1).astype(float) / max(1, right_crop - left_crop)
        max_row = row_ink.max()

        if max_row > 0:
            for r in range(h):
                if row_ink[r] < max_row * 0.60:
                    top_crop = r
                    break
            for r in range(h - 1, -1, -1):
                if row_ink[r] < max_row * 0.60:
                    bot_crop = r + 1
                    break

        if left_crop > w * 0.03:
            print(f"  Phase-2 left crop:  {left_crop}px")
        else:
            left_crop = 0
        if right_crop < w - w * 0.03:
            print(f"  Phase-2 right crop: {w - right_crop}px from right")
        else:
            right_crop = w
        if top_crop > h * 0.03:
            print(f"  Phase-2 top crop:   {top_crop}px")
        else:
            top_crop = 0
        if bot_crop < h - h * 0.03:
            print(f"  Phase-2 bot crop:   {h - bot_crop}px from bottom")
        else:
            bot_crop = h

    img_out = img[top_crop:bot_crop, left_crop:right_crop]
    ox += left_crop
    oy += top_crop

    h_out, w_out = img_out.shape[:2]
    print(f"  Final text area: {w_out}×{h_out} px")

    # Phase 3: upscale if too small
    if h_out < min_upscale_height and h_out > 0:
        scale = min_upscale_height / h_out
        new_w = int(w_out * scale)
        img_out = cv2.resize(img_out, (new_w, min_upscale_height),
                             interpolation=cv2.INTER_CUBIC)
        print(f"  Upscaled ×{scale:.1f} → {img_out.shape[1]}×{img_out.shape[0]} px")

    return img_out, (ox, oy)


# ══════════════════════════════════════════════════════════════════
# STEP 2 — DESKEW
# ══════════════════════════════════════════════════════════════════

def deskew(img: np.ndarray, binary: np.ndarray):
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 500:
        return img, binary, 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5:
        print(f"  Deskew: {angle:.2f}° (skipped)")
        return img, binary, angle
    print(f"  Deskew: correcting {angle:.2f}°")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    bin_out = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_out, bin_out, angle


# ══════════════════════════════════════════════════════════════════
# STEP 3 — SPLIT INTO TEXT PANELS
# ══════════════════════════════════════════════════════════════════

def find_panel_splits(binary: np.ndarray,
                      min_gap_height: int = 30) -> list:
    """Find y-positions of major horizontal dividers between text panels."""
    h = binary.shape[0]
    min_gap_height = max(min_gap_height, int(h * 0.02))

    row_ink = binary.sum(axis=1).astype(float)
    max_ink = row_ink.max()

    if max_ink == 0:
        return [(0, h)]

    gap_thresh = max_ink * 0.03
    is_gap = row_ink < gap_thresh

    dividers = []
    in_gap, start = False, 0
    for i in range(h):
        if is_gap[i] and not in_gap:
            in_gap, start = True, i
        elif not is_gap[i] and in_gap:
            in_gap = False
            if i - start >= min_gap_height:
                dividers.append((start, i))
    if in_gap and (h - start) >= min_gap_height:
        dividers.append((start, h))

    if not dividers:
        return [(0, h)]

    panels = []
    prev = 0
    for gstart, gend in dividers:
        if gstart > prev + 20:
            panels.append((prev, gstart))
        prev = gend
    if prev < h - 20:
        panels.append((prev, h))

    min_panel_h = max(40, int(h * 0.05))
    panels = [(t, b) for t, b in panels if b - t >= min_panel_h]

    if not panels:
        return [(0, h)]

    print(f"  Found {len(panels)} text panel(s): {panels}")
    return panels


# ══════════════════════════════════════════════════════════════════
# STEP 4 — FIND TEXT LINES WITHIN A PANEL (VALLEY-BASED)
# ══════════════════════════════════════════════════════════════════

def find_text_lines_in_panel(binary_panel: np.ndarray,
                              min_line_height: int = 15,
                              valley_threshold: float = None) -> list:
    """Find text line extents via valley detection. Auto-tunes threshold."""
    h = binary_panel.shape[0]
    row_ink = binary_panel.sum(axis=1).astype(float)
    max_ink = row_ink.max()
    if max_ink == 0:
        return []

    if valley_threshold is None:
        sorted_ink = np.sort(row_ink)
        n_valley = max(1, int(h * 0.25))
        median_valley = float(sorted_ink[n_valley // 2])
        valley_threshold = median_valley / max_ink
        valley_threshold = max(0.08, min(0.30, valley_threshold))
        print(f"    Auto valley threshold: {valley_threshold:.2f}")

    thresh_val = max_ink * valley_threshold

    kernel = np.ones(5) / 5
    smooth_ink = np.convolve(row_ink, kernel, mode='same')
    is_valley = smooth_ink < thresh_val

    raw_lines = []
    in_text, start = False, 0
    for i in range(h):
        if not is_valley[i] and not in_text:
            in_text, start = True, i
        elif is_valley[i] and in_text:
            in_text = False
            if i - start >= min_line_height:
                raw_lines.append([start, i])
    if in_text and (h - start) >= min_line_height:
        raw_lines.append([start, h])

    merged = []
    for seg in raw_lines:
        if merged and seg[0] - merged[-1][1] <= 3:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    margin = 3
    return [(max(0, t - margin), min(h, b + margin)) for t, b in merged]


# ══════════════════════════════════════════════════════════════════
# STEP 5 — FIND CHARACTERS IN A LINE
# ══════════════════════════════════════════════════════════════════

def find_characters_in_line(binary_line: np.ndarray,
                             min_w: int = 6,
                             min_h: int = 6,
                             min_area: int = 40,
                             max_aspect: float = 8.0) -> list:
    """Find character bounding boxes via connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_line, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w < min_w or h < min_h or area < min_area:
            continue
        if w / max(h, 1) > max_aspect:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


# ══════════════════════════════════════════════════════════════════
# STEP 6 — MERGE DIACRITICS
# ══════════════════════════════════════════════════════════════════

def merge_diacritics(boxes: list, line_height: int = None) -> list:
    """Merge small diacritic blobs into their base character."""
    if len(boxes) < 2:
        return boxes

    heights = sorted(b[3] for b in boxes)
    median_h = heights[len(heights) // 2]
    dia_thresh = median_h * 0.55

    bases      = [(i, b) for i, b in enumerate(boxes) if b[3] >= dia_thresh]
    diacritics = [(i, b) for i, b in enumerate(boxes) if b[3] < dia_thresh]

    if not diacritics:
        return boxes

    merged = {bi: list(bb) for bi, (_, bb) in enumerate(bases)}
    merged_set = set()

    for di, (_, db) in enumerate(diacritics):
        dx, dy, dw, dh = db
        cx = dx + dw / 2
        best_bi, best_ov = None, -1
        for bi, (_, bb) in enumerate(bases):
            bx, by, bw, bh = bb
            margin = bw * 0.30
            if (bx - margin) <= cx <= (bx + bw + margin):
                ov = min(bx + bw, dx + dw) - max(bx, dx)
                if ov > best_ov:
                    best_ov, best_bi = ov, bi
        if best_bi is not None:
            bx, by, bw, bh = merged[best_bi]
            merged[best_bi] = [
                min(bx, dx),
                min(by, dy),
                max(bx + bw, dx + dw) - min(bx, dx),
                max(by + bh, dy + dh) - min(by, dy),
            ]
            merged_set.add(di)

    final = [tuple(v) for v in merged.values()]
    min_orphan_area = median_h * 5
    for i, (_, db) in enumerate(diacritics):
        if i not in merged_set:
            if db[2] * db[3] >= min_orphan_area:
                final.append(db)

    final.sort(key=lambda b: b[0])
    return final


# ══════════════════════════════════════════════════════════════════
# STEP 7 — CROP & SAVE
# ══════════════════════════════════════════════════════════════════

def crop_and_save(img: np.ndarray,
                  binary: np.ndarray,
                  panels: list,
                  valley_threshold,
                  min_line_height: int,
                  output_dir: str,
                  padding: int = 4,
                  target_size: int = 64) -> list:
    """Crop characters and save to output_dir. Returns metadata list."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata    = []
    total       = 0
    global_line = 0

    for panel_idx, (panel_top, panel_bot) in enumerate(panels):
        panel_bin  = binary[panel_top:panel_bot, :]
        panel_gray = gray[panel_top:panel_bot, :]

        print(f"  Panel {panel_idx}: rows {panel_top}-{panel_bot} "
              f"({panel_bot - panel_top}px)")

        lines = find_text_lines_in_panel(
            panel_bin,
            min_line_height=min_line_height,
            valley_threshold=valley_threshold,
        )
        print(f"    → {len(lines)} lines")

        ih, iw = panel_gray.shape
        for line_top, line_bot in lines:
            line_bin  = panel_bin[line_top:line_bot, :]
            line_gray = panel_gray[line_top:line_bot, :]
            line_h    = line_bot - line_top

            boxes = find_characters_in_line(line_bin)
            boxes = merge_diacritics(boxes, line_height=line_h)
            boxes = [(x, y, w, h) for x, y, w, h in boxes if h <= line_h * 1.5]

            if not boxes:
                global_line += 1
                continue

            for char_idx, (x, y, w, h) in enumerate(boxes):
                x1 = max(0, x - padding)
                x2 = min(iw, x + w + padding)
                y1 = max(0, y - padding)
                y2 = min(ih, y + h + padding)
                crop = line_gray[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                resized  = cv2.resize(crop, (target_size, target_size),
                                      interpolation=cv2.INTER_AREA)
                filename = f"line_{global_line:02d}_char_{char_idx:03d}.png"
                cv2.imwrite(str(out / filename), resized)
                metadata.append({
                    "file":       filename,
                    "line":       global_line,
                    "char_idx":   char_idx,
                    "page_x":     int(x1),
                    "page_y":     int(panel_top + line_top + y1),
                    "width":      int(x2 - x1),
                    "height":     int(y2 - y1),
                    "predicted":  None,
                    "confidence": None,
                    "low_conf":   None,
                    "top5":       None,
                    "bbox": {
                        "x": int(x1),
                        "y": int(panel_top + line_top + y1),
                        "w": int(x2 - x1),
                        "h": int(y2 - y1),
                    },
                })
                total += 1

            print(f"      Line {global_line:02d}: {len(boxes)} chars")
            global_line += 1

    print(f"  Saved {total} crops → {output_dir}/")
    return metadata


# ══════════════════════════════════════════════════════════════════
# DEBUG VISUALISATION
# ══════════════════════════════════════════════════════════════════

def save_debug_image(img: np.ndarray, panels: list,
                     metadata: list, binary: np.ndarray, out_path: str):
    debug = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    w = debug.shape[1]
    for top, bot in panels:
        cv2.rectangle(debug, (0, top), (w, bot), (0, 140, 255), 2)
    for item in metadata:
        bbox = item.get("bbox") or {}
        x = bbox.get("x", item.get("page_x", 0))
        y = bbox.get("y", item.get("page_y", 0))
        bw = bbox.get("w", item.get("width", 10))
        bh = bbox.get("h", item.get("height", 10))
        cv2.rectangle(debug, (x, y), (x + bw, y + bh), (0, 220, 0), 1)
    cv2.imwrite(out_path, debug)
    print(f"  Debug → {out_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def segment_page(image_path: str,
                 output_dir: str          = "output_segments",
                 target_size: int         = 64,
                 debug: bool              = False,
                 min_line_height: int     = 15,
                 valley_threshold: float  = None,
                 min_upscale_height: int  = 600,
                 min_panel_gap: int       = 30) -> list:
    """
    Full segmentation pipeline.

    Parameters
    ----------
    image_path         : path to the manuscript image
    output_dir         : where to save character crops + metadata JSON
    target_size        : resize each crop to this square size (default 64)
    debug              : save debug_segmentation.jpg with boxes drawn
    min_line_height    : discard text runs shorter than this (pixels)
    valley_threshold   : None = auto-tune (recommended). Set 0.08–0.30
                         explicitly only if auto-tuning produces bad splits.
                         CLI flag: --threshold
    min_upscale_height : upscale image if shorter than this (pixels)
    min_panel_gap      : minimum gap height to count as a panel divider

    Returns
    -------
    list of character metadata dicts (same content written to segments_meta.json)
    """
    print(f"\n{'─'*60}")
    print(f"  Segmenting: {Path(image_path).name}")
    print(f"{'─'*60}")

    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    h0, w0 = img_raw.shape[:2]
    print(f"  Loaded: {w0}×{h0} px")

    print("  Cropping borders...")
    img, page_offset = smart_crop(img_raw, min_upscale_height)

    print("  Binarizing...")
    binary = make_binary(img)

    print("  Deskewing...")
    img, binary, angle = deskew(img, binary)

    print("  Finding text panels...")
    panels = find_panel_splits(binary, min_gap_height=min_panel_gap)

    print("  Extracting characters...")
    out = Path(output_dir)
    metadata = crop_and_save(
        img, binary, panels,
        valley_threshold=valley_threshold,
        min_line_height=min_line_height,
        output_dir=output_dir,
        target_size=target_size,
    )

    if not metadata:
        print("\n  ⚠  No characters extracted.")
        print("  Tips:")
        print("    --threshold 0.20   lower valley threshold if lines merge")
        print("    --threshold 0.08   raise if too many false splits")
        print("    --debug            save debug_segmentation.jpg to inspect")
        return []

    meta_file = out / "segments_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_image": str(image_path),
            "deskew_angle": round(angle, 3),
            "num_lines":    max((c["line"] for c in metadata), default=0) + 1,
            "num_chars":    len(metadata),
            "page_offset":  list(page_offset),
            "characters":   metadata,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Metadata → {meta_file}")

    if debug:
        save_debug_image(img, panels, metadata, binary,
                         str(out / "debug_segmentation.jpg"))

    n_lines = max((c["line"] for c in metadata), default=0) + 1
    print(f"\n  ✓ Done: {len(metadata)} characters from {n_lines} lines "
          f"across {len(panels)} panel(s)")
    return metadata


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Newa manuscript segmenter v5")
    p.add_argument("--image",     required=True)
    p.add_argument("--output",    default="output_segments")
    p.add_argument("--size",      type=int,   default=64)
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--threshold", type=float, default=None,
                   help="Valley threshold (0.0–1.0). Default: auto. "
                        "Try 0.20 if lines merge, 0.08 if too many splits.")
    p.add_argument("--min-line-height",    type=int, default=15)
    p.add_argument("--min-upscale-height", type=int, default=600)
    p.add_argument("--min-panel-gap",      type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    segment_page(
        image_path          = args.image,
        output_dir          = args.output,
        target_size         = args.size,
        debug               = args.debug,
        min_line_height     = args.min_line_height,
        valley_threshold    = args.threshold,
        min_upscale_height  = args.min_upscale_height,
        min_panel_gap       = args.min_panel_gap,
    )