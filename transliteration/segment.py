"""
segment.py  —  Newa Manuscript Page Segmentation (v4)
══════════════════════════════════════════════════════════════════

CHANGE LOG
──────────
v1: threshold-based line finding → failed (merged all lines into 1)
v2: better threshold + border crop → still failed (threshold too high)
v3: valley-based + upscaling for low-res → worked for small images
v4: handles HIGH-res manuscript images with decorative borders.
    Key improvements:
    1. Smarter page crop: uses COLUMN profile to strip left/right
       flower/ornament borders (not just dark background).
    2. Detects the decorative band between text panels (red band
       with flowers) and splits the page into separate panels.
    3. Valley threshold auto-tuned per image based on ink statistics.
    4. Upscaling preserved for low-res images (< 600px tall).
    5. All parameters still tunable via CLI flags.

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
# HELPER: COLOUR-AWARE BINARIZATION
# ══════════════════════════════════════════════════════════════════

def make_binary(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image → ink mask (ink=255, background=0).
    Uses B+G average to suppress red/orange ruling lines and decorations.
    """
    b = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    bg = ((b + g) / 2).astype(np.uint8)
    blurred = cv2.GaussianBlur(bg, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )
    binary = cv2.bitwise_or(otsu, adaptive)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — SMART PAGE CROP
# ══════════════════════════════════════════════════════════════════

def smart_crop(img: np.ndarray, min_upscale_height: int = 600):
    """
    Crop out decorative/dark borders on all four sides.

    Two-phase approach:
      Phase 1 (bright-region crop): finds the overall page rectangle
        by looking for the largest bright region. This removes the
        outer dark background (black photo background).
      Phase 2 (column + row profile crop): within the page, removes
        ornamental borders (flower columns, ruled top/bottom bands)
        by finding where text-density ink actually starts and ends.

    Returns: (cropped_img, (x_offset, y_offset))
    """
    h0, w0 = img.shape[:2]

    # ── Phase 1: bright-region crop ──────────────────────────────
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

    # ── Phase 2: strip ornamental column borders ──────────────────
    # Use ink profile in middle 50% of rows to avoid top/bottom margins
    h, w = img.shape[:2]
    binary = make_binary(img)
    mid_binary = binary[h // 4: 3 * h // 4, :]

    col_ink = mid_binary.sum(axis=0).astype(float) / (h // 2)
    max_col = col_ink.max()
    if max_col == 0:
        return img, (ox, oy)

    # A column is "border" if it has > 85% ink coverage AND is on the edge
    border_thresh = max_col * 0.85

    left_crop = 0
    for c in range(w):
        if col_ink[c] < border_thresh:
            left_crop = c
            break

    right_crop = w
    for c in range(w - 1, -1, -1):
        if col_ink[c] < border_thresh:
            right_crop = c + 1
            break

    # Also strip top/bottom: find first/last rows with meaningful ink
    mid_binary2 = binary[:, left_crop:right_crop]
    row_ink = mid_binary2.sum(axis=1).astype(float) / (right_crop - left_crop)
    max_row = row_ink.max()

    top_crop = 0
    for r in range(h):
        if row_ink[r] < max_row * 0.85:
            top_crop = r
            break

    bot_crop = h
    for r in range(h - 1, -1, -1):
        if row_ink[r] < max_row * 0.85:
            bot_crop = r + 1
            break

    # Apply phase-2 crop only if it's meaningful (removes > 2% on any side)
    if left_crop > w * 0.02:
        print(f"  Phase-2 left crop:  {left_crop}px")
    else:
        left_crop = 0
    if right_crop < w - w * 0.02:
        print(f"  Phase-2 right crop: {w - right_crop}px from right")
    else:
        right_crop = w
    if top_crop > h * 0.02:
        print(f"  Phase-2 top crop:   {top_crop}px")
    else:
        top_crop = 0
    if bot_crop < h - h * 0.02:
        print(f"  Phase-2 bot crop:   {h - bot_crop}px from bottom")
    else:
        bot_crop = h

    img_out = img[top_crop:bot_crop, left_crop:right_crop]
    ox += left_crop
    oy += top_crop

    h_out, w_out = img_out.shape[:2]
    print(f"  Final text area: {w_out}×{h_out} px")

    # ── Phase 3: upscale if too small ────────────────────────────
    if h_out < min_upscale_height:
        scale = min_upscale_height / h_out
        img_out = cv2.resize(img_out,
                             (int(w_out * scale), min_upscale_height),
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
    """
    Find y-positions of major horizontal dividers (decorative bands,
    fold lines, header/footer separators).

    Returns list of (top, bottom) row ranges for each panel.
    A major divider is a run of rows where ink drops below 5% of max,
    AND the run is at least min_gap_height pixels tall.

    For manuscripts with a single unbroken text block, returns [(0, h)].
    """
    h = binary.shape[0]
    row_ink = binary.sum(axis=1).astype(float)
    max_ink = row_ink.max()

    if max_ink == 0:
        return [(0, h)]

    # Threshold: < 5% of max = structural gap (not just inter-line gap)
    gap_thresh = max_ink * 0.05
    is_gap = row_ink < gap_thresh

    # Find gap runs >= min_gap_height
    dividers = []
    in_gap = False
    for i in range(h):
        if is_gap[i] and not in_gap:
            in_gap = True; start = i
        elif not is_gap[i] and in_gap:
            in_gap = False
            if i - start >= min_gap_height:
                dividers.append((start, i))
    if in_gap and (h - start) >= min_gap_height:
        dividers.append((start, h))

    if not dividers:
        return [(0, h)]

    # Build panel extents from dividers
    panels = []
    prev = 0
    for gstart, gend in dividers:
        if gstart > prev + 20:  # panel must be at least 20px
            panels.append((prev, gstart))
        prev = gend
    if prev < h - 20:
        panels.append((prev, h))

    print(f"  Found {len(panels)} text panel(s): {panels}")
    return panels


# ══════════════════════════════════════════════════════════════════
# STEP 4 — FIND TEXT LINES WITHIN A PANEL (VALLEY-BASED)
# ══════════════════════════════════════════════════════════════════

def find_text_lines_in_panel(binary_panel: np.ndarray,
                              min_line_height: int = 15,
                              valley_threshold: float = None) -> list:
    """
    Find text line extents within one panel using valley detection.

    valley_threshold: fraction of max_ink below which a row is a valley.
      If None, auto-tuned from the panel's ink statistics:
      - high-res images have more inter-character ink in gap rows
        (because gap rows still contain descenders) → use ~0.20-0.25
      - low-res images have cleaner gaps → use ~0.10-0.15

    Returns list of (top, bottom) relative to panel top.
    """
    h = binary_panel.shape[0]
    row_ink = binary_panel.sum(axis=1).astype(float)
    max_ink = row_ink.max()
    if max_ink == 0:
        return []

    # Auto-tune threshold from gap statistics if not provided
    if valley_threshold is None:
        # Find natural valley values: sort rows, look at bottom 30%
        sorted_ink = np.sort(row_ink)
        n_valley = max(1, int(h * 0.30))  # expect ~30% of rows to be gaps
        median_valley = float(sorted_ink[n_valley // 2])
        valley_threshold = median_valley / max_ink
        # Clamp to reasonable range
        valley_threshold = max(0.05, min(0.35, valley_threshold))
        print(f"    Auto valley threshold: {valley_threshold:.2f}")

    thresh_val = max_ink * valley_threshold
    is_valley = row_ink < thresh_val

    # Smooth with tiny kernel (3px) to avoid single noisy rows
    kernel = np.ones(3) / 3
    smooth_ink = np.convolve(row_ink, kernel, mode='same')
    smooth_thresh = max_ink * valley_threshold
    is_valley = smooth_ink < smooth_thresh

    # Build text runs
    raw_lines = []
    in_text = False
    start   = 0
    for i in range(h):
        if not is_valley[i] and not in_text:
            in_text = True; start = i
        elif is_valley[i] and in_text:
            in_text = False
            if i - start >= min_line_height:
                raw_lines.append([start, i])
    if in_text and (h - start) >= min_line_height:
        raw_lines.append([start, h])

    # Merge lines separated by very narrow valleys (≤ 4px — descenders)
    merged = []
    for seg in raw_lines:
        if merged and seg[0] - merged[-1][1] <= 4:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    # Add margin
    margin = 4
    result = [(max(0, t - margin), min(h, b + margin)) for t, b in merged]
    return result


# ══════════════════════════════════════════════════════════════════
# STEP 5 — FIND CHARACTERS IN A LINE
# ══════════════════════════════════════════════════════════════════

def find_characters_in_line(binary_line: np.ndarray,
                             min_w: int = 5,
                             min_h: int = 5,
                             min_area: int = 15) -> list:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_line, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w >= min_w and h >= min_h and area >= min_area:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes


# ══════════════════════════════════════════════════════════════════
# STEP 6 — MERGE DIACRITICS
# ══════════════════════════════════════════════════════════════════

def merge_diacritics(boxes: list) -> list:
    if len(boxes) < 2:
        return boxes
    heights    = sorted(b[3] for b in boxes)
    median_h   = heights[len(heights) // 2]
    dia_thresh = median_h * 0.55
    bases      = [(i, b) for i, b in enumerate(boxes) if b[3] >= dia_thresh]
    diacritics = [(i, b) for i, b in enumerate(boxes) if b[3] < dia_thresh]
    if not diacritics:
        return boxes
    merged     = {bi: list(bb) for bi, (_, bb) in enumerate(bases)}
    merged_set = set()
    for di, (_, db) in enumerate(diacritics):
        dx, dy, dw, dh = db
        cx = dx + dw / 2
        best_bi, best_ov = None, 0
        for bi, (_, bb) in enumerate(bases):
            bx, by, bw, bh = bb
            if bx <= cx <= bx + bw:
                ov = min(bx + bw, dx + dw) - max(bx, dx)
                if ov > best_ov:
                    best_ov, best_bi = ov, bi
        if best_bi is not None:
            bx, by, bw, bh = merged[best_bi]
            merged[best_bi] = [min(bx, dx), min(by, dy),
                               max(bx+bw, dx+dw)-min(bx, dx),
                               max(by+bh, dy+dh)-min(by, dy)]
            merged_set.add(di)
    final = [tuple(v) for v in merged.values()]
    final += [db for i, (_, db) in enumerate(diacritics) if i not in merged_set]
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
                  padding: int = 5,
                  target_size: int = 64) -> list:
    """
    For each panel, find lines, find characters, crop and save.
    Line indices are global across panels (line 0, 1, 2 ... N).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata  = []
    total     = 0
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

            boxes = find_characters_in_line(line_bin)
            boxes = merge_diacritics(boxes)

            if not boxes:
                global_line += 1
                continue

            for char_idx, (x, y, w, h) in enumerate(boxes):
                x1 = max(0, x - padding);  x2 = min(iw, x + w + padding)
                y1 = max(0, y - padding);  y2 = min(ih, y + h + padding)
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
                    "predicted":  None, "confidence": None,
                    "low_conf":   None, "top5":       None,
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
        cv2.rectangle(debug, (0, top), (w, bot), (200, 100, 0), 2)

    for item in metadata:
        x, y, bw, bh = item["page_x"], item["page_y"], item["width"], item["height"]
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

    valley_threshold: None = auto-tune per image (recommended).
      Set explicitly (e.g. 0.20) only if auto-tuning fails.
    min_panel_gap: minimum height (px) to be considered a structural
      divider between panels (default 30).
    """
    print(f"\n{'─'*60}")
    print(f"  Segmenting: {Path(image_path).name}")
    print(f"{'─'*60}")

    img_raw = cv2.imread(image_path)
    if img_raw is None:
        raise FileNotFoundError(f"Cannot open: {image_path}")
    h0, w0 = img_raw.shape[:2]
    print(f"  Loaded: {w0}×{h0} px")

    # 1. Smart crop
    print("  Cropping borders...")
    img, page_offset = smart_crop(img_raw, min_upscale_height)

    # 2. Binarize
    print("  Binarizing...")
    binary = make_binary(img)

    # 3. Deskew
    print("  Deskewing...")
    img, binary, angle = deskew(img, binary)

    # 4. Find panels
    print("  Finding text panels...")
    panels = find_panel_splits(binary, min_gap_height=min_panel_gap)

    # 5. Extract characters
    print("  Extracting characters...")
    out = Path(output_dir)
    metadata = crop_and_save(img, binary, panels,
                             valley_threshold=valley_threshold,
                             min_line_height=min_line_height,
                             output_dir=output_dir,
                             target_size=target_size)

    if not metadata:
        print("\n  ⚠  No characters extracted.")
        print("  Tips:")
        print("    --threshold 0.20   lower if lines are merged")
        print("    --threshold 0.08   raise if too many false splits")
        print("    --debug            save debug_segmentation.jpg to inspect")
        return []

    # 6. Save metadata
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

    # 7. Debug
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
    p = argparse.ArgumentParser(description="Newa manuscript segmenter v4")
    p.add_argument("--image",     required=True)
    p.add_argument("--output",    default="output_segments")
    p.add_argument("--size",      type=int,   default=64)
    p.add_argument("--debug",     action="store_true")
    p.add_argument("--threshold", type=float, default=None,
                   help="Valley threshold (0.0–1.0). Default: auto. "
                        "Try 0.20 if lines merge, 0.08 if too many splits.")
    p.add_argument("--min-line-height", type=int, default=15)
    p.add_argument("--min-upscale-height", type=int, default=600)
    p.add_argument("--min-panel-gap", type=int, default=30,
                   help="Min pixel height to count as a panel divider (default 30)")
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