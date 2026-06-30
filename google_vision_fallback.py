"""
google_vision_fallback.py  —  Google Vision ONLY mode
═══════════════════════════════════════════════════════
All recognition goes through Google Cloud Vision.
CNN model is not used at all.

Set your API key before running:
  Windows PS:   $env:GOOGLE_VISION_API_KEY = "AIza..."
  Linux/macOS:  export GOOGLE_VISION_API_KEY="AIza..."
"""

import base64
import io
import json
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# Pillow 10.x uses Image.Resampling enum
_RESAMPLE = Image.Resampling.LANCZOS

# ── API key ───────────────────────────────────────────────────────
GOOGLE_VISION_API_KEY: str = os.environ.get("GOOGLE_VISION_API_KEY", "")

# These are unused but kept so app.py imports don't break
VISION_ATTEMPT_THRESHOLD: float = 0.0   # always use Vision
BLANK_THRESHOLD: float = 0.20

# ── Devanagari → class name (reverse of NEWA_TO_DEVA in app.py) ───
_DEVA_TO_CLASS: dict = {
    "क": "ka",   "ख": "kha",  "ग": "ga",   "घ": "gha",  "ङ": "nga",
    "च": "ca",   "छ": "cha",  "ज": "ja",   "झ": "jha",  "ञ": "nya",
    "ट": "tta",  "ठ": "ttha", "ड": "dda",  "ढ": "ddha", "ण": "nna",
    "त": "ta",   "थ": "tha",  "द": "da",   "ध": "dha",  "न": "na",
    "प": "pa",   "फ": "pha",  "ब": "ba",   "भ": "bha",  "म": "ma",
    "य": "ya",   "र": "ra",   "ल": "la",   "व": "wa",   "स": "sa",
    "श": "sha",  "ष": "ssa",  "ह": "ha",
    "अ": "vowel_A",  "आ": "vowel_AA", "इ": "vowel_I",  "ई": "vowel_II",
    "उ": "vowel_U",  "ऊ": "vowel_UU", "ए": "vowel_E",  "ऐ": "vowel_AI",
    "ओ": "vowel_O",  "औ": "vowel_AU",
    "ा": "matra_aa", "ि": "matra_i",  "ी": "matra_ii",
    "ु": "matra_u",  "ू": "matra_uu", "े": "matra_e",
    "ै": "matra_ai", "ो": "matra_o",  "ौ": "matra_au",
    "ं": "anusvara", "ः": "visarga",  "ँ": "candrabindu",
    "्": "virama",   "ऽ": "avagraha",
    "०": "digit_0",  "१": "digit_1",  "२": "digit_2",  "३": "digit_3",
    "४": "digit_4",  "५": "digit_5",  "६": "digit_6",  "७": "digit_7",
    "८": "digit_8",  "९": "digit_9",
}


def _pil_to_b64(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_vision_api(img_b64: str) -> Tuple[str, str]:
    """
    Returns (detected_text, error_message).
    On success error_message is "".
    On failure detected_text is "".
    """
    if not GOOGLE_VISION_API_KEY:
        return "", "GOOGLE_VISION_API_KEY not set"

    payload = json.dumps({
        "requests": [{
            "image": {"content": img_b64},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 5}],
            "imageContext": {"languageHints": ["ne", "sa"]},
        }]
    }).encode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        block = data.get("responses", [{}])[0]
        if "error" in block:
            err = block["error"]
            msg = f"API error {err.get('code')}: {err.get('message')}"
            print(f"[Vision] {msg}")
            return "", msg

        annotations = block.get("textAnnotations", [])
        if not annotations:
            print("[Vision] No text detected in crop")
            return "", ""

        text = annotations[0].get("description", "").strip()
        print(f"[Vision] Raw detected text: {repr(text)}")
        return text, ""

    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
            msg = body.get("error", {}).get("message", str(e))
        except Exception:
            msg = str(e)
        print(f"[Vision] HTTP {e.code}: {msg}")
        return "", f"HTTP {e.code}: {msg}"

    except urllib.error.URLError as e:
        print(f"[Vision] Network error: {e.reason}")
        return "", f"Network: {e.reason}"

    except Exception as e:
        print(f"[Vision] Exception: {e}")
        return "", str(e)


def _text_to_class(text: str) -> Optional[str]:
    """Map Devanagari text from Vision back to internal class name."""
    if not text:
        return None
    # Strip whitespace and newlines Vision sometimes adds
    text = text.strip().replace("\n", "").replace(" ", "")
    if not text:
        return None
    # Try two-char match first (consonant + matra fused)
    if len(text) >= 2 and text[:2] in _DEVA_TO_CLASS:
        return _DEVA_TO_CLASS[text[:2]]
    # Single char
    if text[0] in _DEVA_TO_CLASS:
        return _DEVA_TO_CLASS[text[0]]
    print(f"[Vision] Cannot map to class: {repr(text)}")
    return None


def _prepare_image(img_array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL, upscale if too small for Vision."""
    if isinstance(img_array, np.ndarray):
        pil = Image.fromarray(img_array).convert("RGB")
    else:
        pil = img_array.convert("RGB")

    w, h = pil.size
    # Vision needs at least ~64px to read characters reliably
    if w < 64 or h < 64:
        scale = max(64 / w, 64 / h)
        new_w, new_h = max(64, int(w * scale)), max(64, int(h * scale))
        pil = pil.resize((new_w, new_h), _RESAMPLE)
        print(f"[Vision] Upscaled crop {w}x{h} → {new_w}x{new_h}")

    return pil


def google_vision_recognise(
    img_array: np.ndarray,
) -> Tuple[Optional[str], float, str]:
    """
    Recognise a character crop using Google Cloud Vision ONLY.

    Returns:
        (class_name_or_None, confidence, source_label)
    """
    if not GOOGLE_VISION_API_KEY:
        print("[Vision] ERROR: GOOGLE_VISION_API_KEY is not set!")
        return None, 0.0, "no_api_key"

    try:
        pil = _prepare_image(img_array)
        img_b64 = _pil_to_b64(pil)
        raw_text, error = _call_vision_api(img_b64)

        if error:
            return None, 0.0, "vision_error"

        if not raw_text:
            return None, 0.0, "vision_no_text"

        class_name = _text_to_class(raw_text)
        if class_name:
            print(f"[Vision] Mapped '{raw_text}' → {class_name}")
            return class_name, 0.90, "google_vision"

        return None, 0.0, "vision_unmapped"

    except Exception as e:
        print(f"[Vision] Unhandled exception: {e}")
        return None, 0.0, "vision_exception"


def should_use_fallback(confidence: float) -> bool:
    """Always True — every character goes through Vision."""
    return True


def is_truly_blank(confidence: float) -> bool:
    return confidence < BLANK_THRESHOLD


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC — run directly to test
# Usage:  python google_vision_fallback.py [crop.png]
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  Google Vision — standalone diagnostic")
    print("=" * 55)

    if not GOOGLE_VISION_API_KEY:
        print("\nERROR: GOOGLE_VISION_API_KEY is not set.\n")
        print("  Windows PS:   $env:GOOGLE_VISION_API_KEY = 'AIza...'")
        print("  Linux/macOS:  export GOOGLE_VISION_API_KEY='AIza...'")
        sys.exit(1)

    key_preview = GOOGLE_VISION_API_KEY[:8] + "..." + GOOGLE_VISION_API_KEY[-4:]
    print(f"\nAPI key : {key_preview}")
    print(f"Pillow  : {Image.__version__}")
    print(f"Resample: {_RESAMPLE}\n")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Image: {path}")
        img = np.array(Image.open(path).convert("RGB"))
        print(f"Size : {img.shape[1]}x{img.shape[0]}\n")
        class_name, conf, source = google_vision_recognise(img)
        print(f"\nResult  : class={class_name}  conf={conf:.0%}  source={source}")
    else:
        print("No image supplied — sending a white 64x64 test image...")
        blank = np.full((64, 64, 3), 255, dtype=np.uint8)
        class_name, conf, source = google_vision_recognise(blank)
        print(f"\nResult  : class={class_name}  conf={conf:.0%}  source={source}")
        print("\n(White image returns no text — that is correct)")
        print("Pass a real crop to test recognition:")
        print("  python google_vision_fallback.py path/to/crop.png")