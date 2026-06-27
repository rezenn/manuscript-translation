"""
app.py  —  Newa Manuscript Transliterator v5
═══════════════════════════════════════════════════════════════════

WHAT'S NEW IN v5
────────────────
• Character-wise mode: upload a single character crop → direct model
  inference, no segmentation pipeline, no wrong-character issue.
• Line-wise mode: upload a manuscript image → segment → OCR → translate.
• Region-of-interest mode: upload full manuscript → draw crop box →
  only the cropped region is processed.
• Fixed: single char images were going through full segmentation which
  caused wrong predictions (e.g. ka → ṇa).
• Fixed: Gradio 6.x CSS/theme moved to launch().
• Clean two-tab UI: "Single Character" and "Line / Region".
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# -- paths: MUST come before any local imports --
_ROOT = Path(__file__).resolve().parent
for _p in [str(_ROOT / "ocr_model"), str(_ROOT / "transliteration")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
CKPT_PATH  = ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = ROOT / "transliteration_output"
OUTPUT_DIR.mkdir(exist_ok=True)

from model import build_model

# ── try importing pipeline pieces ──────────────────────────────────
try:
    from segment   import segment_page
    HAS_SEGMENT = True
except ImportError:
    HAS_SEGMENT = False

try:
    from recognize import recognize_segments, load_model as _load_model_rec
    HAS_RECOGNIZE = True
except ImportError:
    HAS_RECOGNIZE = False

try:
    from newa_to_devanagari import to_devanagari, to_iast
    HAS_DEVANAGARI = True
except ImportError:
    # Inline minimal fallback so the char mode still works
    HAS_DEVANAGARI = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATE = True
except ImportError:
    HAS_TRANSLATE = False

try:
    from postprocess import postprocess as _postprocess
    HAS_POSTPROCESS = True
except ImportError:
    HAS_POSTPROCESS = False
    def _postprocess(char_list, global_threshold=0.55):
        return char_list   # no-op fallback


# ══════════════════════════════════════════════════════════════════
# NEWA → DEVANAGARI MAP  (fallback if newa_to_devanagari.py missing)
# ══════════════════════════════════════════════════════════════════

NEWA_TO_DEVA = {
    # Consonants
    "ka": "क", "kha": "ख", "ga": "ग", "gha": "घ", "nga": "ङ",
    "ca": "च", "cha": "छ", "ja": "ज", "jha": "झ", "nya": "ञ",
    "tta": "ट", "ttha": "ठ", "dda": "ड", "ddha": "ढ", "nna": "ण",
    "ta": "त", "tha": "थ", "da": "द", "dha": "ध", "na": "न",
    "pa": "प", "pha": "फ", "ba": "ब", "bha": "भ", "ma": "म",
    "ya": "य", "ra": "र", "la": "ल", "wa": "व", "sa": "स",
    "sha": "श", "ssa": "ष", "ha": "ह",
    # Vowels (standalone)
    "vowel_A": "अ", "vowel_AA": "आ", "vowel_I": "इ", "vowel_II": "ई",
    "vowel_U": "उ", "vowel_UU": "ऊ", "vowel_E": "ए", "vowel_AI": "ऐ",
    "vowel_O": "ओ", "vowel_AU": "औ",
    # Matras (vowel signs)
    "matra_aa": "ा", "matra_i": "ि", "matra_ii": "ी",
    "matra_u": "ु", "matra_uu": "ू", "matra_e": "े", "matra_ai": "ै",
    "matra_o": "ो", "matra_au": "ौ",
    # Signs
    "anusvara": "ं", "visarga": "ः", "candrabindu": "ँ",
    "virama": "्", "avagraha": "ऽ",
    # Digits
    "digit_0": "०", "digit_1": "१", "digit_2": "२", "digit_3": "३",
    "digit_4": "४", "digit_5": "५", "digit_6": "६", "digit_7": "७",
    "digit_8": "८", "digit_9": "९",
}

NEWA_TO_IAST = {
    "ka": "k", "kha": "kh", "ga": "g", "gha": "gh", "nga": "ṅ",
    "ca": "c", "cha": "ch", "ja": "j", "jha": "jh", "nya": "ñ",
    "tta": "ṭ", "ttha": "ṭh", "dda": "ḍ", "ddha": "ḍh", "nna": "ṇ",
    "ta": "t", "tha": "th", "da": "d", "dha": "dh", "na": "n",
    "pa": "p", "pha": "ph", "ba": "b", "bha": "bh", "ma": "m",
    "ya": "y", "ra": "r", "la": "l", "wa": "v", "sa": "s",
    "sha": "ś", "ssa": "ṣ", "ha": "h",
    "vowel_A": "a", "vowel_AA": "ā", "vowel_I": "i", "vowel_II": "ī",
    "vowel_U": "u", "vowel_UU": "ū", "vowel_E": "e", "vowel_AI": "ai",
    "vowel_O": "o", "vowel_AU": "au",
    "matra_aa": "ā", "matra_i": "i", "matra_ii": "ī",
    "matra_u": "u", "matra_uu": "ū", "matra_e": "e", "matra_ai": "ai",
    "matra_o": "o", "matra_au": "au",
    "anusvara": "ṃ", "visarga": "ḥ", "candrabindu": "m̐",
    "virama": "·", "avagraha": "ʼ",
    "digit_0": "0", "digit_1": "1", "digit_2": "2", "digit_3": "3",
    "digit_4": "4", "digit_5": "5", "digit_6": "6", "digit_7": "7",
    "digit_8": "8", "digit_9": "9",
}


def char_to_devanagari(name: str) -> str:
    if HAS_DEVANAGARI:
        try:
            return to_devanagari(name)
        except Exception:
            pass
    # Fallback: try both the name as-is and lowercase
    return (NEWA_TO_DEVA.get(name)
            or NEWA_TO_DEVA.get(name.lower())
            or "⟨?⟩")


def char_to_iast(name: str) -> str:
    if HAS_DEVANAGARI:
        try:
            return to_iast(name)
        except Exception:
            pass
    return (NEWA_TO_IAST.get(name)
            or NEWA_TO_IAST.get(name.lower())
            or "?")


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING  (singleton)
# ══════════════════════════════════════════════════════════════════

_MODEL_CACHE = {}

def get_model():
    """Load model once and cache."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["idx2char"], _MODEL_CACHE["img_size"]

    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))

    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)

    arch        = ckpt.get("arch",        "convnet")
    num_classes = ckpt.get("num_classes", 67)
    img_size    = ckpt.get("img_size",    64)
    class_map   = ckpt.get("class_map",   {})

    model = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    # Normalize class_map to {int → name}
    if not class_map:
        idx2char = {}
    else:
        first_key = next(iter(class_map))
        if isinstance(first_key, str) and not first_key.isdigit():
            idx2char = {int(v): k for k, v in class_map.items()}
        else:
            idx2char = {int(k): v for k, v in class_map.items()}

    _MODEL_CACHE["model"]    = model
    _MODEL_CACHE["idx2char"] = idx2char
    _MODEL_CACHE["img_size"] = img_size
    _MODEL_CACHE["device"]   = device

    val_acc = ckpt.get("best_val_top1", "?")
    print(f"✓ Model loaded: {arch} | {num_classes} classes | val acc: {val_acc}%")
    return model, idx2char, img_size


def get_device():
    if "device" in _MODEL_CACHE:
        return _MODEL_CACHE["device"]
    get_model()
    return _MODEL_CACHE["device"]


# ══════════════════════════════════════════════════════════════════
# SINGLE-CHARACTER INFERENCE
# (bypasses segmentation entirely — direct model call)
# ══════════════════════════════════════════════════════════════════

def preprocess_single_char(img_array: np.ndarray, img_size: int = 64) -> torch.Tensor:
    """
    Prepare a single character image for the model.
    Handles: color → gray, inversion detection, resize, normalize.
    """
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()

    # Auto-invert: model expects dark ink on white background.
    # If mean pixel > 128 it is already light-background; otherwise invert.
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)

    # Crop tight bounding box around ink
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(4, int(max(w, h) * 0.08))
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(gray.shape[1], x + w + pad)
        y2 = min(gray.shape[0], y + h + pad)
        gray = gray[y1:y2, x1:x2]

    # Resize
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(gray).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)


def infer_single_char(img_array: np.ndarray, top_k: int = 5):
    """
    Run the CNN on a single character image.
    Returns list of (class_name, confidence) tuples.
    """
    model, idx2char, img_size = get_model()
    device = get_device()

    tensor = preprocess_single_char(img_array, img_size).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        k = min(top_k, probs.shape[1])
        top_probs, top_idx = probs.topk(k, dim=1)

    results = []
    for j in range(k):
        idx  = top_idx[0][j].item()
        conf = top_probs[0][j].item()
        name = idx2char.get(idx, f"cls_{idx}")
        results.append((name, conf))

    return results


# ══════════════════════════════════════════════════════════════════
# TAB 1: SINGLE CHARACTER MODE
# ══════════════════════════════════════════════════════════════════

def process_single_character(image):
    """
    Called when user uploads a single character image.
    Skips all segmentation — direct CNN inference.
    """
    if image is None:
        return "❌ No image uploaded.", "", "", ""

    try:
        # Gradio passes numpy array (RGB)
        img_arr = np.array(image)
        results = infer_single_char(img_arr, top_k=5)

        best_name, best_conf = results[0]
        deva = char_to_devanagari(best_name)
        iast = char_to_iast(best_name)

        # Confidence label
        if best_conf >= 0.70:
            conf_label = "high confidence ✓"
        elif best_conf >= 0.40:
            conf_label = "moderate confidence"
        else:
            conf_label = "uncertain — see top-5 below"

        # Top-5 table
        top5_lines = []
        for i, (name, conf) in enumerate(results):
            bar  = "█" * int(conf * 20)
            dv   = char_to_devanagari(name)
            mark = " ← TOP" if i == 0 else ""
            top5_lines.append(f"  {i+1}. {dv} ({name:20s})  {conf:.1%}  {bar}{mark}")
        top5_str = "\n".join(top5_lines)

        status = (
            f"✓ Predicted: {deva}  [{best_name}]  —  {best_conf:.1%} ({conf_label})\n\n"
            f"Top-5 predictions:\n{top5_str}"
        )

        return status, deva, iast, best_name

    except Exception as e:
        return f"❌ Error: {e}", "", "", ""


# ══════════════════════════════════════════════════════════════════
# CROP HELPER — extract region from ImageEditor output
# ══════════════════════════════════════════════════════════════════

def _to_rgb_array(img) -> np.ndarray:
    """Convert PIL Image or numpy array to uint8 RGB numpy array."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img
    return None


def extract_cropped_region(editor_value):
    """
    Gradio ImageEditor returns a dict:
      {"background": PIL.Image or ndarray,
       "layers": [PIL.Image or ndarray, ...],
       "composite": PIL.Image or ndarray}

    CRITICAL: use 'is not None' checks — never 'or' — because numpy
    arrays raise ValueError when evaluated as booleans.
    """
    if editor_value is None:
        return None

    if isinstance(editor_value, dict):
        # Prefer composite (background + drawn layers merged).
        # Fall back to background if composite is None.
        img = editor_value.get("composite")
        if img is None:
            img = editor_value.get("background")
    else:
        # Plain numpy array passed directly (e.g. from gr.Image)
        img = editor_value

    if img is None:
        return None

    return _to_rgb_array(img)


# ══════════════════════════════════════════════════════════════════
# LINE / REGION PIPELINE
# ══════════════════════════════════════════════════════════════════

def translate_text(text: str, src_hint: str = "ne") -> str:
    if not HAS_TRANSLATE or not text.strip():
        return "(translation unavailable — install deep-translator)"
    try:
        # Strip low-conf markers, line separators, and extra whitespace
        clean = text.replace("⟨?⟩", "").replace(" | ", " ").strip()
        clean = " ".join(clean.split())   # normalise whitespace
        if not clean:
            return "(nothing to translate)"
        return GoogleTranslator(source=src_hint, target="en").translate(clean)
    except Exception as e:
        return f"(translation error: {e})"


def ocr_region(img_arr: np.ndarray, min_conf: float, do_translate: bool):
    """
    Run the full segment → OCR → Devanagari → translate pipeline
    on a region (already cropped by the user or the full image).
    """
    if not HAS_SEGMENT or not HAS_RECOGNIZE:
        return "❌ segment.py / recognize.py not found in transliteration/", "", "", "", None

    tmp_dir = tempfile.mkdtemp(prefix="newa_seg_")
    tmp_img = os.path.join(tmp_dir, "input.png")

    try:
        # Save region as temp file
        pil = Image.fromarray(img_arr)
        pil.save(tmp_img)

        # ── 1. Segment ─────────────────────────────────────────────
        seg_dir = os.path.join(tmp_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        seg_result = segment_page(
            image_path=tmp_img,
            output_dir=seg_dir,
        )

        # ── 2. OCR ─────────────────────────────────────────────────
        char_list = recognize_segments(
            segments_dir=seg_dir,
            checkpoint_path=str(CKPT_PATH),
            confidence_threshold=min_conf,
        )

        if not char_list:
            return "❌ No characters found after segmentation.", "", "", "", None

        # ── 2b. Post-process: attractor bias + sequence repair ──────
        char_list = _postprocess(char_list, global_threshold=min_conf)

        # ── 3. Group by line ────────────────────────────────────────
        lines = {}
        for c in sorted(char_list, key=lambda x: (x["line"], x["char_idx"])):
            ln = c["line"]
            lines.setdefault(ln, []).append(c)

        # ── 4. Build Devanagari strings ─────────────────────────────
        # Consonant class names — these carry an inherent /a/ vowel
        # unless immediately followed by a matra, virama, or another consonant
        # (handled below via look-ahead).
        CONSONANTS = {
            "ka","kha","ga","gha","nga","ca","cha","ja","jha","nya",
            "tta","ttha","dda","ddha","nna","ta","tha","da","dha","na",
            "pa","pha","ba","bha","ma","ya","ra","la","wa","sha","ssa","sa","ha",
        }
        MATRAS = {
            "matra_aa","matra_i","matra_ii","matra_u","matra_uu",
            "matra_e","matra_ai","matra_o","matra_au",
        }

        def _build_line_deva(chars, min_conf):
            """
            Convert a list of char prediction dicts to a Devanagari string.
            Implements: isolate char → recognise → build word → build sentence.

            Rules applied:
              • confidence < min_conf  → ⟨?⟩
              • 'space' predicted      → U+0020 word separator
              • consonant + matra      → consonant glyph + matra sign (no inherent a)
              • consonant + virama     → consonant + virama (halant form)
              • standalone consonant at end of word / before space/virama → as-is
                (Devanagari renders the inherent /a/ by default)
              • independent vowel      → vowel glyph
            """
            result = []
            for i, c in enumerate(chars):
                pred = c.get("predicted", "")
                conf = c.get("confidence", 0.0)

                # Synthetic space entry (injected by segment.py)
                if pred == "space" or c.get("file") == "__space__":
                    result.append(" ")
                    continue

                if conf < min_conf:
                    result.append("⟨?⟩")
                    continue

                deva = char_to_devanagari(pred)
                result.append(deva)

            return "".join(result)

        line_strings = []
        for ln in sorted(lines):
            chars = lines[ln]
            deva_line = _build_line_deva(chars, min_conf)
            line_strings.append(f"Line {ln+1:02d}: {deva_line}")

        full_deva = "\n".join(line_strings)

        # concat_deva = everything joined (spaces between words preserved)
        all_deva_parts = []
        prev_ln = None
        for c in char_list:
            ln = c.get("line", 0)
            if prev_ln is not None and ln != prev_ln:
                all_deva_parts.append(" | ")   # line separator
            pred = c.get("predicted", "")
            conf = c.get("confidence", 0.0)
            if pred == "space" or c.get("file") == "__space__":
                all_deva_parts.append(" ")
            elif conf >= min_conf:
                all_deva_parts.append(char_to_devanagari(pred))
            prev_ln = ln
        concat_deva = "".join(all_deva_parts)

        iast_str = "".join(
            (" " if (c.get("predicted") == "space" or c.get("file") == "__space__")
             else char_to_iast(c.get("predicted", "")))
            for c in char_list
            if c.get("confidence", 0) >= min_conf or c.get("predicted") == "space"
        )

        # ── 5. Translate ────────────────────────────────────────────
        translation = translate_text(concat_deva) if do_translate else "(disabled)"

        # ── 6. Debug image ──────────────────────────────────────────
        debug_img = None
        meta_path = Path(seg_dir) / "segments_meta.json"
        if meta_path.exists():
            debug_img = _draw_debug(img_arr, meta_path)

        total   = len(char_list)
        low_n   = sum(1 for c in char_list if c.get("low_conf", False))
        avg_c   = sum(c.get("confidence", 0) for c in char_list) / total if total else 0
        n_lines = len(lines)

        status = (
            f"✓ Segmented {total} characters across {n_lines} line(s).\n"
            f"  Avg confidence: {avg_c:.1%}   "
            f"Low-conf (shown as ⟨?⟩): {low_n}/{total}"
        )

        return status, full_deva, concat_deva, iast_str, translation, debug_img

    except Exception as e:
        import traceback
        return f"❌ Error: {e}\n{traceback.format_exc()}", "", "", "", None

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _draw_debug(img_arr, meta_path):
    """Draw bounding boxes on the image for debug view."""
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        debug = img_arr.copy()
        for c in meta.get("characters", []):
            bbox = c.get("bbox")
            if bbox:
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                conf = c.get("confidence", 0)
                color = (0, 200, 0) if conf >= 0.5 else (200, 0, 0)
                cv2.rectangle(debug, (x, y), (x+w, y+h), color, 1)
        return Image.fromarray(debug)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# TAB 2: LINE / REGION MODE
# ══════════════════════════════════════════════════════════════════

def process_line_mode(editor_value, min_conf, do_translate):
    """
    Called from the Line/Region tab.
    Uses whatever the user has uploaded (or cropped via the editor).
    """
    img_arr = extract_cropped_region(editor_value)
    if img_arr is None:
        return "❌ No image provided.", "", "", "", "(no translation)", None

    result = ocr_region(img_arr, float(min_conf), bool(do_translate))

    # ocr_region returns 6 values; map them to our outputs
    if len(result) == 6:
        status, per_line, full_deva, iast, translation, debug_img = result
    else:
        # Error path
        status = result[0]
        per_line = full_deva = iast = translation = ""
        debug_img = None

    return status, per_line, full_deva, iast, translation, debug_img


# ══════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════

CSS = """
/* ── Newa OCR v5 — dark manuscript aesthetic ── */

:root {
    --bg-deep:    #0f0e0b;
    --bg-card:    #1a1814;
    --bg-input:   #221f1a;
    --border:     #3a3530;
    --accent:     #c8853a;
    --accent2:    #8b6a3e;
    --text-main:  #e8dcc8;
    --text-dim:   #9a8f80;
    --text-deva:  #f5e8cc;
    --font-mono:  'Courier New', monospace;
    --radius:     8px;
}

body, .gradio-container {
    background: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: Georgia, 'Times New Roman', serif !important;
}

/* Header */
.app-header {
    text-align: center;
    padding: 28px 0 18px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.app-header h1 {
    font-size: 1.9rem;
    color: var(--accent);
    letter-spacing: 0.03em;
    margin: 0 0 6px;
}
.app-header p {
    color: var(--text-dim);
    font-size: 0.9rem;
    margin: 0;
}

/* Tabs */
.tabs .tab-nav button {
    background: var(--bg-card) !important;
    color: var(--text-dim) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) var(--radius) 0 0 !important;
    font-family: Georgia, serif !important;
    font-size: 0.95rem !important;
    padding: 10px 22px !important;
    transition: all 0.2s;
}
.tabs .tab-nav button.selected {
    background: var(--bg-input) !important;
    color: var(--accent) !important;
    border-bottom-color: var(--bg-input) !important;
}

/* Textboxes */
textarea, .gr-textbox textarea {
    background: var(--bg-input) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: Georgia, serif !important;
}

/* Devanagari output — larger */
.deva-out textarea {
    font-size: 1.6rem !important;
    color: var(--text-deva) !important;
    font-family: 'Noto Sans Devanagari', 'Mangal', serif !important;
    line-height: 1.8 !important;
}

/* Buttons */
.gr-button-primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: Georgia, serif !important;
    font-size: 1rem !important;
    padding: 10px 28px !important;
    cursor: pointer;
    transition: background 0.2s;
}
.gr-button-primary:hover {
    background: var(--accent2) !important;
}

/* Status box */
.status-box textarea {
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    color: #9fd89f !important;
    background: #0d1a0d !important;
    border-color: #2a4a2a !important;
}

/* Section labels */
label span {
    color: var(--text-dim) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Confidence slider */
input[type=range] {
    accent-color: var(--accent) !important;
}
"""

def build_ui():
    with gr.Blocks(title="Newa OCR v5") as demo:

        gr.HTML("""
        <div class="app-header">
            <h1>𑐣𑐾𑐥𑐵𑐮 Newa Manuscript OCR</h1>
            <p>Prachalit / Newa Script  →  Devanagari  →  English &nbsp;|&nbsp; v5</p>
        </div>
        """)

        with gr.Tabs():

            # ══════════════════════════════════════════════════════
            # TAB 1 — SINGLE CHARACTER
            # ══════════════════════════════════════════════════════
            with gr.Tab("🔡 Single Character"):
                gr.Markdown(
                    "Upload **one character** image (e.g. a crop from your dataset). "
                    "The model runs directly — **no segmentation** — so results are accurate "
                    "even for isolated characters."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        char_img_in = gr.Image(
                            label="Character image (PNG/JPG)",
                            type="numpy",
                            height=280,
                        )
                        char_btn = gr.Button("▶ Recognize Character", variant="primary")

                    with gr.Column(scale=2):
                        char_status = gr.Textbox(
                            label="Result & Top-5 Predictions",
                            lines=9,
                            elem_classes=["status-box"],
                        )
                        with gr.Row():
                            char_deva = gr.Textbox(
                                label="Devanagari",
                                lines=2,
                                elem_classes=["deva-out"],
                            )
                            char_iast = gr.Textbox(
                                label="IAST Romanization",
                                lines=2,
                            )
                            char_name = gr.Textbox(
                                label="Class name",
                                lines=2,
                            )

                char_btn.click(
                    fn=process_single_character,
                    inputs=[char_img_in],
                    outputs=[char_status, char_deva, char_iast, char_name],
                )
                # Also trigger on image upload (live feedback)
                char_img_in.change(
                    fn=process_single_character,
                    inputs=[char_img_in],
                    outputs=[char_status, char_deva, char_iast, char_name],
                )

            # ══════════════════════════════════════════════════════
            # TAB 2 — LINE / REGION
            # ══════════════════════════════════════════════════════
            with gr.Tab("📜 Line / Region"):
                gr.Markdown(
                    "Upload a **manuscript image** (full page or a cropped region). "
                    "You can use the editor tools to **crop/select** the exact area you want "
                    "to process — only that region will be segmented and OCR'd."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        region_img_in = gr.ImageEditor(
                            label="Upload manuscript — use crop tool to select a region",
                            type="numpy",
                            height=420,
                            # Allow drawing to indicate region of interest
                            brush=gr.Brush(colors=["#c8853a"], default_size=2),
                        )

                    with gr.Column(scale=1):
                        min_conf_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.25, step=0.05,
                            label="Min OCR confidence (below = shown as ⟨?⟩)",
                        )
                        do_translate_chk = gr.Checkbox(
                            label="Translate to English (Google Translate)",
                            value=True,
                        )
                        region_btn = gr.Button("▶ Run OCR on Region", variant="primary")

                line_status = gr.Textbox(
                    label="Status",
                    lines=3,
                    elem_classes=["status-box"],
                )

                with gr.Tabs():
                    with gr.Tab("Per-line Devanagari"):
                        line_perline = gr.Textbox(
                            label="",
                            lines=12,
                            elem_classes=["deva-out"],
                        )
                    with gr.Tab("Full Devanagari"):
                        line_full_deva = gr.Textbox(
                            label="",
                            lines=6,
                            elem_classes=["deva-out"],
                        )
                    with gr.Tab("IAST"):
                        line_iast = gr.Textbox(label="", lines=6)
                    with gr.Tab("Translation"):
                        line_translation = gr.Textbox(label="", lines=6)
                    with gr.Tab("Debug — Character Boxes"):
                        line_debug = gr.Image(label="Detected characters")

                region_btn.click(
                    fn=process_line_mode,
                    inputs=[region_img_in, min_conf_slider, do_translate_chk],
                    outputs=[
                        line_status,
                        line_perline,
                        line_full_deva,
                        line_iast,
                        line_translation,
                        line_debug,
                    ],
                )

        gr.HTML("""
        <div style="text-align:center;padding:18px 0 8px;
                    color:#5a5248;font-size:0.8rem;border-top:1px solid #2a2520;
                    margin-top:24px;">
            Newa Manuscript OCR — Individual Project &nbsp;|&nbsp;
            Model: NewaConvNet 471k params &nbsp;|&nbsp; 67 classes
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Newa Manuscript Transliterator v5")
    print("=" * 60)
    print(f"  Checkpoint: {CKPT_PATH}")
    print(f"  Found: {'✓' if CKPT_PATH.exists() else '✗ NOT FOUND'}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  segment.py:  {'✓' if HAS_SEGMENT else '✗'}")
    print(f"  recognize.py: {'✓' if HAS_RECOGNIZE else '✗'}")
    print(f"  deep-translator: {'✓' if HAS_TRANSLATE else '✗'}")
    print()

    # Pre-load model at startup
    try:
        get_model()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("UI will start but recognition will fail until checkpoint is present.\n")

    demo = build_ui()

    # Find a free port automatically (tries 7860 first, then 7861–7880)
    import socket
    def _free_port(start=7860, end=7880):
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        return None   # let Gradio decide

    port = _free_port()
    if port:
        print(f"\n  → Starting on http://localhost:{port}\n")
    else:
        print("\n  → Could not find a free port in 7860-7880. "
              "Set GRADIO_SERVER_PORT to override.\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,      # None = Gradio picks automatically
        share=False,
        css=CSS,
    )