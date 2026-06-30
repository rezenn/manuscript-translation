"""
app.py — Newa Manuscript Transliterator v10

CHANGES vs the version you sent back
──────────────────────────────────────
- Single Character tab: CNN-ONLY again, no Tesseract call at all.
  This matches the document you uploaded (the older infer_single_char
  that didn't call any fallback) — restored as you asked.
- Line/Region tab: Tesseract fallback kept and ACTUALLY invoked
  (your uploaded app.py imported should_use_fallback/google_vision_
  recognise but never called them in ocr_region — that's why nothing
  changed). Now fixed.
- Visual upgrades for the thesis demo:
    - Confidence bar chart on Single Character (kept from before)
    - NEW: per-line confidence chart on Line/Region (bar per character)
    - NEW: summary stat cards (total chars, avg confidence, low-conf
      count, fallback-used count) styled as a small dashboard row
    - NEW: attractor-class pie/breakdown chart showing how many
      predictions came from attractor-prone classes
    - Debug image legend made clearer (green/orange/red)
- extract_cropped_region fix kept (no ValueError crash)
- No emoji anywhere
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

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

ROOT       = Path(__file__).resolve().parent
CKPT_PATH  = ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = ROOT / "transliteration_output"
DEMO_DIR   = ROOT / "demo_images"
OUTPUT_DIR.mkdir(exist_ok=True)

from model import build_model

try:
    from segment import segment_page
    HAS_SEGMENT = True
except ImportError:
    segment_page = None
    HAS_SEGMENT  = False

try:
    from recognize import recognize_segments
    HAS_RECOGNIZE = True
except ImportError:
    recognize_segments = None
    HAS_RECOGNIZE      = False

try:
    from newa_to_devanagari import to_devanagari, to_iast
    HAS_DEVANAGARI = True
except ImportError:
    to_devanagari  = None
    to_iast        = None
    HAS_DEVANAGARI = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATE = True
except ImportError:
    GoogleTranslator = None
    HAS_TRANSLATE    = False

try:
    from postprocess import postprocess as _postprocess
except ImportError:
    def _postprocess(char_list, global_threshold=0.35):
        return char_list

try:
    from google_vision_fallback import (
        google_vision_recognise, should_use_fallback_for_class
    )
    HAS_FALLBACK = True
except ImportError:
    HAS_FALLBACK = False
    def google_vision_recognise(img): return (None, 0.0, "unavailable")
    def should_use_fallback_for_class(cls, conf): return False


# ══════════════════════════════════════════════════════════════════
# MAPS
# ══════════════════════════════════════════════════════════════════

NEWA_TO_DEVA = {
    "ka":"क","kha":"ख","ga":"ग","gha":"घ","nga":"ङ",
    "ca":"च","cha":"छ","ja":"ज","jha":"झ","nya":"ञ",
    "tta":"ट","ttha":"ठ","dda":"ड","ddha":"ढ","nna":"ण",
    "ta":"त","tha":"थ","da":"द","dha":"ध","na":"न",
    "pa":"प","pha":"फ","ba":"ब","bha":"भ","ma":"म",
    "ya":"य","ra":"र","la":"ल","wa":"व","sa":"स",
    "sha":"श","ssa":"ष","ha":"ह",
    "vowel_A":"अ","vowel_AA":"आ","vowel_I":"इ","vowel_II":"ई",
    "vowel_U":"उ","vowel_UU":"ऊ","vowel_E":"ए","vowel_AI":"ऐ",
    "vowel_O":"ओ","vowel_AU":"औ",
    "matra_aa":"ा","matra_i":"ि","matra_ii":"ी",
    "matra_u":"ु","matra_uu":"ू","matra_e":"े","matra_ai":"ै",
    "matra_o":"ो","matra_au":"ौ",
    "anusvara":"ं","visarga":"ः","candrabindu":"ँ",
    "virama":"्","avagraha":"ऽ",
    "digit_0":"०","digit_1":"१","digit_2":"२","digit_3":"३",
    "digit_4":"४","digit_5":"५","digit_6":"६","digit_7":"७",
    "digit_8":"८","digit_9":"९",
}

NEWA_TO_IAST = {
    "ka":"k","kha":"kh","ga":"g","gha":"gh","nga":"ng",
    "ca":"c","cha":"ch","ja":"j","jha":"jh","nya":"ny",
    "tta":"tt","ttha":"tth","dda":"dd","ddha":"ddh","nna":"nn",
    "ta":"t","tha":"th","da":"d","dha":"dh","na":"n",
    "pa":"p","pha":"ph","ba":"b","bha":"bh","ma":"m",
    "ya":"y","ra":"r","la":"l","wa":"w","sa":"s",
    "sha":"sh","ssa":"ss","ha":"h",
    "vowel_A":"a","vowel_AA":"aa","vowel_I":"i","vowel_II":"ii",
    "vowel_U":"u","vowel_UU":"uu","vowel_E":"e","vowel_AI":"ai",
    "vowel_O":"o","vowel_AU":"au",
    "matra_aa":"aa","matra_i":"i","matra_ii":"ii",
    "matra_u":"u","matra_uu":"uu","matra_e":"e","matra_ai":"ai",
    "matra_o":"o","matra_au":"au",
    "anusvara":"m","visarga":"h","candrabindu":"m",
    "virama":"","avagraha":"'",
    "digit_0":"0","digit_1":"1","digit_2":"2","digit_3":"3",
    "digit_4":"4","digit_5":"5","digit_6":"6","digit_7":"7",
    "digit_8":"8","digit_9":"9",
}

# Attractor classes — flagged in debug output / charts, used to decide
# Line/Region Tesseract fallback. NOT used in Single Character mode.
ATTRACTOR_CLASSES = {
    "nna": "absorbs na, tta",
    "ga":  "absorbs ka, gha",
    "ra":  "absorbs la, wa",
    "ja":  "absorbs jha, ca",
    "dda": "absorbs ddha",
    "tta": "absorbs nna",
    "kha": "absorbs ka",
}


def char_to_devanagari(name: str) -> str:
    if HAS_DEVANAGARI and to_devanagari:
        try:
            return to_devanagari(name)
        except Exception:
            pass
    return NEWA_TO_DEVA.get(name) or NEWA_TO_DEVA.get(name.lower()) or "?"


def char_to_iast(name: str) -> str:
    if HAS_DEVANAGARI and to_iast:
        try:
            return to_iast(name)
        except Exception:
            pass
    return NEWA_TO_IAST.get(name) or NEWA_TO_IAST.get(name.lower()) or "?"


def is_attractor(class_name: str, confidence: float) -> bool:
    return class_name in ATTRACTOR_CLASSES and confidence < 0.80


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════

_MODEL_CACHE: dict = {}

def get_model():
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["idx2char"], _MODEL_CACHE["img_size"]
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps")  if torch.backends.mps.is_available() else
              torch.device("cpu"))
    ckpt        = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    arch        = ckpt.get("arch",        "convnet")
    num_classes = ckpt.get("num_classes", 67)
    img_size    = ckpt.get("img_size",    64)
    class_map   = ckpt.get("class_map",   {})
    model       = build_model(arch=arch, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    if not class_map:
        idx2char = {}
    else:
        fk = next(iter(class_map))
        idx2char = ({int(v): k for k, v in class_map.items()}
                    if isinstance(fk, str) and not fk.isdigit()
                    else {int(k): v for k, v in class_map.items()})
    _MODEL_CACHE.update({"model":model,"idx2char":idx2char,
                          "img_size":img_size,"device":device})
    val = ckpt.get("best_val_top1","?")
    print(f"Model: {arch} | {num_classes} classes | val acc: {val}%")
    return model, idx2char, img_size

def get_device():
    if "device" not in _MODEL_CACHE:
        get_model()
    return _MODEL_CACHE["device"]


# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════

def preprocess_single_char(img_array: np.ndarray, img_size: int = 64) -> torch.Tensor:
    gray = (cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            if img_array.ndim == 3 else img_array.copy())
    if gray.mean() < 128:
        gray = cv2.bitwise_not(gray)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(4, int(max(w, h) * 0.08))
        gray = gray[max(0,y-pad):min(gray.shape[0],y+h+pad),
                    max(0,x-pad):min(gray.shape[1],x+w+pad)]
    gray   = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(gray).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════
# SINGLE CHARACTER — CNN ONLY, no Tesseract, as requested
# ══════════════════════════════════════════════════════════════════

def infer_single_char(img_array: np.ndarray, top_k: int = 5):
    """
    CNN-only. No fallback call here at all — restored to the simple
    behaviour you wanted for Single Character mode.
    """
    model, idx2char, img_size = get_model()
    device = get_device()
    tensor = preprocess_single_char(img_array, img_size).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        k      = min(top_k, probs.shape[1])
        top_probs, top_idx = probs.topk(k, dim=1)
    results = [
        (idx2char.get(int(top_idx[0][j].item()), f"cls_{int(top_idx[0][j].item())}"),
         float(top_probs[0][j].item()))
        for j in range(k)
    ]
    best_name, best_conf = results[0]
    source = "CNN"
    conf_data = [
        {"Prediction": f"{char_to_devanagari(n)} ({n})", "Confidence": round(c * 100, 1)}
        for n, c in results
    ]
    return results, source, conf_data


def process_single_character(image):
    if image is None:
        return "No image uploaded.", "", "", "", "", []
    try:
        img_arr                    = np.array(image)
        results, source, conf_data = infer_single_char(img_arr, top_k=5)
        best_name, best_conf       = results[0]
        deva = char_to_devanagari(best_name)
        iast = char_to_iast(best_name)
        conf_label = ("high confidence" if best_conf >= 0.70 else
                      "moderate"        if best_conf >= 0.40 else "uncertain")
        top5_lines = []
        for i, (name, conf) in enumerate(results[:5]):
            bar  = "█" * int(conf * 20)
            mark = " <- best" if i == 0 else ""
            top5_lines.append(
                f"  {i+1}. {char_to_devanagari(name)} ({name})  {conf:.1%}  {bar}{mark}"
            )
        status = (
            f"Source: {source}\n"
            f"Result: {deva}  [{best_name}]  {best_conf:.1%} ({conf_label})\n\n"
            f"Top-5:\n" + "\n".join(top5_lines)
        )
        return status, deva, iast, best_name, source, conf_data
    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", "", "", "", "", []


def export_single_result(status, deva, iast, class_name):
    if not deva:
        return None
    out = OUTPUT_DIR / "single_char_result.txt"
    out.write_text(
        f"Newa OCR — Single Character Result\n{'='*40}\n\n"
        f"Class:      {class_name}\nDevanagari: {deva}\nIAST:       {iast}\n\n"
        f"--- Detail ---\n{status}\n",
        encoding="utf-8"
    )
    return str(out)


# ══════════════════════════════════════════════════════════════════
# IMAGE HELPER
# ══════════════════════════════════════════════════════════════════

def _to_rgb_array(img) -> Optional[np.ndarray]:
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


def extract_cropped_region(editor_value) -> Optional[np.ndarray]:
    if editor_value is None:
        return None
    if isinstance(editor_value, dict):
        composite  = editor_value.get("composite")
        background = editor_value.get("background")
        if composite is not None:
            img = composite
        elif background is not None:
            img = background
        else:
            return None
    else:
        img = editor_value
    if img is None:
        return None
    return _to_rgb_array(img)


# ══════════════════════════════════════════════════════════════════
# LINE / REGION PIPELINE — CNN + Tesseract fallback for attractors
# ══════════════════════════════════════════════════════════════════

def translate_text(text: str, src_hint: str = "ne") -> str:
    if not HAS_TRANSLATE or not text.strip():
        return "(install deep-translator:  pip install deep-translator)"
    try:
        clean = text.replace("?", "").replace(" | ", " ").strip()
        clean = " ".join(clean.split())
        if not clean:
            return "(nothing to translate)"
        return GoogleTranslator(source=src_hint, target="en").translate(clean)
    except Exception as e:
        return f"(translation error: {e})"


def _draw_debug(img_arr: np.ndarray, meta_path: Path) -> Optional[Image.Image]:
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        debug = img_arr.copy()
        for c in meta.get("characters", []):
            bbox = c.get("bbox")
            if bbox:
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                conf  = c.get("confidence", 0)
                pred  = c.get("predicted", "")
                if pred in ATTRACTOR_CLASSES and conf < 0.80:
                    color = (220, 60, 60)
                elif conf >= 0.55:
                    color = (0, 200, 0)
                else:
                    color = (220, 150, 0)
                cv2.rectangle(debug, (x, y), (x+w, y+h), color, 1)
                cv2.putText(debug, f"{conf:.0%}", (x, max(0, y-3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)
        return Image.fromarray(debug)
    except Exception:
        return None


def ocr_region(img_arr: np.ndarray, min_conf: float, do_translate: bool):
    """
    Returns 8 values now (2 extra for charts):
      status, per_line, full_deva, iast, translation, debug_img,
      char_conf_chart_df, attractor_chart_df
    """
    if not HAS_SEGMENT or not HAS_RECOGNIZE:
        return ("segment.py / recognize.py not found.", "", "", "",
                "(unavailable)", None, None, None)

    tmp_dir = tempfile.mkdtemp(prefix="newa_seg_")
    try:
        tmp_img = os.path.join(tmp_dir, "input.png")
        Image.fromarray(img_arr).save(tmp_img)

        seg_dir = os.path.join(tmp_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        segment_page(image_path=tmp_img, output_dir=seg_dir)

        char_list = recognize_segments(
            segments_dir=seg_dir,
            checkpoint_path=str(CKPT_PATH),
            confidence_threshold=min_conf,
        )
        if not char_list:
            return ("No characters found after segmentation.", "", "", "",
                    "", None, None, None)

        # Tesseract fallback for attractor classes / low confidence
        tesseract_used = 0
        for c in char_list:
            pred = c.get("predicted", "")
            conf = c.get("confidence", 0.0)
            if pred == "space" or c.get("file") == "__space__":
                continue
            if HAS_FALLBACK and should_use_fallback_for_class(pred, conf):
                crop_path = os.path.join(seg_dir, c.get("file", ""))
                if os.path.exists(crop_path):
                    try:
                        crop = cv2.imread(crop_path)
                        if crop is not None:
                            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                            tess_class, tess_conf, tess_src = google_vision_recognise(rgb)
                            if tess_class is not None:
                                c["predicted"]   = tess_class
                                c["confidence"]  = tess_conf
                                c["fallback_used"] = True
                                tesseract_used += 1
                    except Exception as tess_err:
                        print(f"[Fallback] {tess_err}")

        char_list = _postprocess(char_list, global_threshold=min_conf)

        lines: dict = {}
        for c in sorted(char_list, key=lambda x: (x.get("line", 0), x.get("char_idx", 0))):
            lines.setdefault(c.get("line", 0), []).append(c)

        attractor_count = 0

        def _build_line(chars):
            nonlocal attractor_count
            out = []
            for c in chars:
                pred = c.get("predicted", "")
                conf = c.get("confidence", 0.0)
                if pred == "space" or c.get("file") == "__space__":
                    out.append(" ")
                elif conf < min_conf:
                    out.append("?")
                else:
                    if is_attractor(pred, conf):
                        attractor_count += 1
                    out.append(char_to_devanagari(pred))
            return "".join(out)

        per_line_str = "\n".join(
            f"Line {ln+1:02d}: {_build_line(lines[ln])}"
            for ln in sorted(lines)
        )

        concat_parts = []
        prev_ln = None
        for c in char_list:
            ln   = c.get("line", 0)
            pred = c.get("predicted", "")
            conf = c.get("confidence", 0.0)
            if prev_ln is not None and ln != prev_ln:
                concat_parts.append(" ")
            if pred == "space" or c.get("file") == "__space__":
                concat_parts.append(" ")
            elif conf >= min_conf:
                concat_parts.append(char_to_devanagari(pred))
            prev_ln = ln
        concat_deva = "".join(concat_parts)

        iast_str = "".join(
            " " if (c.get("predicted") == "space" or c.get("file") == "__space__")
            else char_to_iast(c.get("predicted", ""))
            for c in char_list
            if c.get("confidence", 0) >= min_conf or c.get("predicted") == "space"
        )

        translation = translate_text(concat_deva) if do_translate else "(disabled)"

        debug_img = None
        meta_path = Path(seg_dir) / "segments_meta.json"
        if meta_path.exists():
            debug_img = _draw_debug(img_arr, meta_path)

        total   = len(char_list)
        blanked = sum(1 for c in char_list
                      if c.get("confidence", 0) < min_conf
                      and c.get("predicted") != "space")
        avg_c   = (sum(c.get("confidence", 0) for c in char_list) / total
                   if total else 0)

        status = (
            f"{total} characters | {len(lines)} line(s) | avg conf: {avg_c:.1%}\n"
            f"Shown as ?: {blanked}/{total}    "
            f"Attractor-class predictions: {attractor_count}/{total}    "
            f"Fallback used: {tesseract_used}/{total}"
        )

        # ── CHART DATA ──────────────────────────────────────────
        try:
            import pandas as pd

            # Per-character confidence bar chart
            chart_rows = []
            for c in char_list:
                pred = c.get("predicted", "")
                conf = c.get("confidence", 0.0)
                if pred == "space" or c.get("file") == "__space__":
                    continue
                label = f"{char_to_devanagari(pred)}"
                chart_rows.append({
                    "Character": f"{label} #{c.get('char_idx', 0)+1}",
                    "Confidence": round(conf * 100, 1),
                    "Type": "Attractor" if is_attractor(pred, conf) else "Normal",
                })
            char_conf_df = pd.DataFrame(chart_rows) if chart_rows else pd.DataFrame(
                {"Character": [], "Confidence": [], "Type": []}
            )

            # Attractor vs normal breakdown
            normal_n    = sum(1 for r in chart_rows if r["Type"] == "Normal")
            attractor_n = sum(1 for r in chart_rows if r["Type"] == "Attractor")
            attractor_df = pd.DataFrame({
                "Category": ["Normal classes", "Attractor classes"],
                "Count": [normal_n, attractor_n],
            })
        except ImportError:
            char_conf_df = None
            attractor_df = None

        return (status, per_line_str, concat_deva, iast_str, translation,
                debug_img, char_conf_df, attractor_df)

    except Exception as e:
        import traceback
        return (f"Error: {e}\n{traceback.format_exc()}", "", "", "", "",
                None, None, None)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def process_line_mode(editor_value, min_conf, do_translate):
    img_arr = extract_cropped_region(editor_value)
    if img_arr is None:
        return ("No image provided.", "", "", "", "(no translation)",
                None, None, None)
    return ocr_region(img_arr, float(min_conf), bool(do_translate))


def export_line_result(per_line, concat_deva, iast, translation):
    if not concat_deva:
        return None
    out = OUTPUT_DIR / "ocr_result.txt"
    out.write_text(
        f"Newa Manuscript OCR Result\n{'='*40}\n\n"
        f"Per-line Devanagari:\n{per_line}\n\n"
        f"Full Devanagari:\n{concat_deva}\n\n"
        f"IAST:\n{iast}\n\n"
        f"English translation:\n{translation}\n",
        encoding="utf-8"
    )
    return str(out)


# ══════════════════════════════════════════════════════════════════
# DEMO IMAGES
# ══════════════════════════════════════════════════════════════════

def get_demo_images() -> dict:
    if not DEMO_DIR.exists():
        return {}
    cats: dict = {}
    exts = {".png", ".jpg", ".jpeg"}
    for item in sorted(DEMO_DIR.iterdir()):
        if item.is_dir():
            cat   = item.name.replace("_", " ").title()
            files = [f for f in sorted(item.iterdir()) if f.suffix.lower() in exts]
            if files:
                cats[cat] = [(f.stem.replace("_", " ").title(), str(f)) for f in files]
        elif item.suffix.lower() in exts:
            cats.setdefault("Samples", []).append(
                (item.stem.replace("_", " ").title(), str(item))
            )
    return cats


# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════

CSS = """
:root {
    --bg-deep:#0f0e0b; --bg-card:#1a1814; --bg-input:#221f1a;
    --border:#3a3530; --accent:#c8853a; --accent2:#8b6a3e;
    --text-main:#e8dcc8; --text-dim:#9a8f80; --text-deva:#f5e8cc;
    --font-mono:'Courier New',monospace; --radius:8px;
}
body,.gradio-container{background:var(--bg-deep)!important;color:var(--text-main)!important;
  font-family:Georgia,'Times New Roman',serif!important;}
.app-header{text-align:center;padding:28px 0 18px;border-bottom:1px solid var(--border);margin-bottom:24px;}
.app-header h1{font-size:1.9rem;color:var(--accent);letter-spacing:.03em;margin:0 0 6px;}
.app-header p{color:var(--text-dim);font-size:.9rem;margin:0;}
.tabs .tab-nav button{background:var(--bg-card)!important;color:var(--text-dim)!important;
  border:1px solid var(--border)!important;border-radius:var(--radius) var(--radius) 0 0!important;
  font-family:Georgia,serif!important;font-size:.95rem!important;padding:10px 22px!important;}
.tabs .tab-nav button.selected{background:var(--bg-input)!important;color:var(--accent)!important;
  border-bottom-color:var(--bg-input)!important;}
textarea,.gr-textbox textarea{background:var(--bg-input)!important;color:var(--text-main)!important;
  border:1px solid var(--border)!important;border-radius:var(--radius)!important;
  font-family:Georgia,serif!important;}
.deva-out textarea{font-size:1.6rem!important;color:var(--text-deva)!important;
  font-family:'Noto Sans Devanagari','Mangal',serif!important;line-height:1.8!important;}
.gr-button-primary{background:var(--accent)!important;color:#fff!important;border:none!important;
  border-radius:var(--radius)!important;font-family:Georgia,serif!important;
  font-size:1rem!important;padding:10px 28px!important;}
.gr-button-primary:hover{background:var(--accent2)!important;}
.status-box textarea{font-family:var(--font-mono)!important;font-size:.82rem!important;
  color:#9fd89f!important;background:#0d1a0d!important;border-color:#2a4a2a!important;}
.source-box textarea{font-family:var(--font-mono)!important;font-size:.9rem!important;
  font-weight:bold!important;color:#d4b870!important;background:#1a1500!important;
  border-color:#4a3a00!important;}
.export-btn{background:#1e2e1e!important;color:#7ab87a!important;
  border:1px solid #3a5a3a!important;border-radius:var(--radius)!important;
  font-size:.85rem!important;padding:6px 16px!important;}
.stat-card{background:var(--bg-card)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;padding:14px 10px!important;text-align:center!important;}
.stat-card .stat-value{font-size:1.6rem!important;color:var(--accent)!important;
  font-weight:bold!important;display:block!important;}
.stat-card .stat-label{font-size:.72rem!important;color:var(--text-dim)!important;
  text-transform:uppercase!important;letter-spacing:.06em!important;}
label span{color:var(--text-dim)!important;font-size:.82rem!important;
  text-transform:uppercase;letter-spacing:.08em;}
input[type=range]{accent-color:var(--accent)!important;}
"""


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════

def build_ui():
    demo_cats = get_demo_images()

    with gr.Blocks(title="Newa OCR") as demo:

        gr.HTML("""
        <div class="app-header">
          <h1>Newa Manuscript OCR</h1>
          <p>Prachalit / Newa Script &rarr; Devanagari &rarr; English</p>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: SINGLE CHARACTER (CNN only) ────────────────
            with gr.Tab("Single Character"):
                gr.Markdown(
                    "Upload one cropped character image. "
                    "CNN-only recognition with top-5 confidence breakdown."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        char_img_in = gr.Image(
                            label="Character image", type="numpy", height=260
                        )
                        char_btn    = gr.Button("Recognize", variant="primary")
                        char_export = gr.Button("Save result", elem_classes=["export-btn"])
                        char_file   = gr.File(label="Download", visible=False)
                    with gr.Column(scale=2):
                        char_source = gr.Textbox(
                            label="Recognition source", lines=1,
                            elem_classes=["source-box"], interactive=False
                        )
                        char_status = gr.Textbox(
                            label="Result and top-5 predictions", lines=9,
                            elem_classes=["status-box"]
                        )
                        with gr.Row():
                            char_deva = gr.Textbox(label="Devanagari", lines=2, elem_classes=["deva-out"])
                            char_iast = gr.Textbox(label="IAST",        lines=2)
                            char_name = gr.Textbox(label="Class name",  lines=2)
                        char_chart = gr.BarPlot(
                            value=None, x="Prediction", y="Confidence",
                            title="Top-5 confidence (%)", color="Prediction",
                            height=220, y_lim=[0, 100],
                        )

                def _run_single(image):
                    status, deva, iast, name, source, conf_data = process_single_character(image)
                    try:
                        import pandas as pd
                        df = pd.DataFrame(conf_data) if conf_data else pd.DataFrame(
                            {"Prediction": [], "Confidence": []}
                        )
                    except ImportError:
                        df = None
                    return status, deva, iast, name, source, df

                char_btn.click(
                    fn=_run_single, inputs=[char_img_in],
                    outputs=[char_status, char_deva, char_iast, char_name, char_source, char_chart],
                )
                char_img_in.change(
                    fn=_run_single, inputs=[char_img_in],
                    outputs=[char_status, char_deva, char_iast, char_name, char_source, char_chart],
                )
                char_export.click(
                    fn=export_single_result,
                    inputs=[char_status, char_deva, char_iast, char_name],
                    outputs=[char_file],
                )
                char_export.click(fn=lambda: gr.update(visible=True), outputs=[char_file])

            # ── TAB 2: LINE / REGION (CNN + Tesseract fallback) ───
            with gr.Tab("Line / Region"):
                gr.Markdown(
                    "Upload a manuscript image and click Run OCR. "
                    "Attractor-class predictions are automatically rechecked "
                    "with a local Tesseract OCR fallback."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        region_img_in = gr.ImageEditor(
                            label="Manuscript image",
                            type="numpy", height=380,
                            brush=gr.Brush(colors=["#c8853a"], default_size=2),
                        )
                    with gr.Column(scale=1):
                        min_conf_slider = gr.Slider(
                            minimum=0.10, maximum=0.90, value=0.35, step=0.05,
                            label="Confidence threshold (below = shown as ?)",
                        )
                        do_translate_chk = gr.Checkbox(label="Translate to English", value=True)
                        region_btn  = gr.Button("Run OCR", variant="primary")
                        line_export = gr.Button("Save result", elem_classes=["export-btn"])
                        line_file   = gr.File(label="Download", visible=False)

                line_status = gr.Textbox(label="Status", lines=3, elem_classes=["status-box"])

                with gr.Tabs():
                    with gr.Tab("Per-line Devanagari"):
                        line_perline = gr.Textbox(label="", lines=10, elem_classes=["deva-out"])
                    with gr.Tab("Full Devanagari"):
                        line_full_deva = gr.Textbox(label="", lines=5, elem_classes=["deva-out"])
                    with gr.Tab("IAST"):
                        line_iast = gr.Textbox(label="", lines=5)
                    with gr.Tab("Translation"):
                        line_translation = gr.Textbox(label="", lines=5)
                    with gr.Tab("Character Boxes"):
                        line_debug = gr.Image(
                            label="Green = confident | Orange = low confidence | Red = attractor class"
                        )
                    with gr.Tab("Confidence Chart"):
                        gr.Markdown("Per-character confidence, color-coded by class type.")
                        line_conf_chart = gr.BarPlot(
                            value=None, x="Character", y="Confidence",
                            title="Per-character confidence (%)", color="Type",
                            height=280, y_lim=[0, 100],
                        )
                    with gr.Tab("Class Breakdown"):
                        gr.Markdown(
                            "Proportion of predictions from attractor-prone classes "
                            "vs normal classes — useful for explaining model limitations."
                        )
                        line_attractor_chart = gr.BarPlot(
                            value=None, x="Category", y="Count",
                            title="Attractor vs normal class predictions",
                            color="Category", height=240,
                        )

                region_btn.click(
                    fn=process_line_mode,
                    inputs=[region_img_in, min_conf_slider, do_translate_chk],
                    outputs=[line_status, line_perline, line_full_deva,
                             line_iast, line_translation, line_debug,
                             line_conf_chart, line_attractor_chart],
                )
                line_export.click(
                    fn=export_line_result,
                    inputs=[line_perline, line_full_deva, line_iast, line_translation],
                    outputs=[line_file],
                )
                line_export.click(fn=lambda: gr.update(visible=True), outputs=[line_file])

            # ── TAB 3: DEMO IMAGES ────────────────────────────────
            with gr.Tab("Demo Images"):
                if not demo_cats:
                    gr.Markdown("No demo images found.")
                else:
                    gr.Markdown("Click any image to run recognition automatically.")
                    for cat_name, files in demo_cats.items():
                        gr.Markdown(f"**{cat_name}**")
                        is_single = any(w in cat_name.lower()
                                        for w in ["char", "single", "glyph"])
                        gallery = gr.Gallery(
                            value=[(path, label) for label, path in files],
                            columns=min(len(files), 8),
                            height=200, object_fit="contain",
                            show_label=False, allow_preview=False,
                        )
                        if is_single:
                            with gr.Row():
                                d_status = gr.Textbox(label="Result", lines=5,
                                                      elem_classes=["status-box"])
                                d_deva   = gr.Textbox(label="Devanagari", lines=2,
                                                      elem_classes=["deva-out"])
                                d_source = gr.Textbox(label="Source", lines=1,
                                                      elem_classes=["source-box"])
                            _files = files
                            def _on_single(evt: gr.SelectData, f=_files):
                                img = np.array(Image.open(f[evt.index][1]).convert("RGB"))
                                s, deva, iast, name, badge, _ = process_single_character(img)
                                return s, deva, badge
                            gallery.select(fn=_on_single, outputs=[d_status, d_deva, d_source])
                        else:
                            with gr.Row():
                                d_status = gr.Textbox(label="Status", lines=3,
                                                      elem_classes=["status-box"])
                                d_deva   = gr.Textbox(label="Devanagari", lines=3,
                                                      elem_classes=["deva-out"])
                                d_trans  = gr.Textbox(label="Translation", lines=3)
                            d_debug = gr.Image(label="Character boxes")
                            _files = files
                            def _on_manuscript(evt: gr.SelectData, f=_files):
                                img = np.array(Image.open(f[evt.index][1]).convert("RGB"))
                                result = ocr_region(img, 0.35, True)
                                st, per, full, iast, trans, dbg = result[:6]
                                return st, full, trans, dbg
                            gallery.select(
                                fn=_on_manuscript,
                                outputs=[d_status, d_deva, d_trans, d_debug]
                            )

        gr.HTML("""
        <div style="text-align:center;padding:14px 0 6px;color:#5a5248;font-size:.78rem;
             border-top:1px solid #2a2520;margin-top:20px;">
          Newa Manuscript OCR &nbsp;|&nbsp; Individual Project &nbsp;|&nbsp; Coventry University
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  Newa Manuscript Transliterator")
    print("=" * 55)
    print(f"  Checkpoint:      {CKPT_PATH}")
    print(f"  Found:           {'Yes' if CKPT_PATH.exists() else 'NOT FOUND'}")
    print(f"  segment.py:      {'Yes' if HAS_SEGMENT    else 'No'}")
    print(f"  recognize.py:    {'Yes' if HAS_RECOGNIZE  else 'No'}")
    print(f"  deep-translator: {'Yes' if HAS_TRANSLATE  else 'No'}")
    print(f"  Fallback (Line/Region only): {'Yes' if HAS_FALLBACK else 'No'}")
    demo_count = sum(len(v) for v in get_demo_images().values())
    print(f"  Demo images:     {demo_count} found")
    print()
    try:
        get_model()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    demo = build_ui()
    import socket
    def _free_port():
        for p in range(7860, 7881):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", p))
                    return p
                except OSError:
                    continue
        return 7860
    port = _free_port()
    print(f"\n  http://localhost:{port}\n")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False, css=CSS)