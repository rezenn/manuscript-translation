"""
google_vision_fallback.py  —  Tesseract OCR fallback (local, free, no billing)

WHY THIS CHANGED
─────────────────
Google Cloud Vision requires a billing-enabled GCP project even on the
free tier (confirmed by your HTTP 403 "requires billing" error). With
no card available this is a dead end for your deadline.

Tesseract OCR is the correct replacement:
  - Fully open source, runs 100% locally
  - No API key, no billing, no internet required after install
  - Has an official Devanagari (Hindi) trained model that reads
    consonants, matras, and conjuncts reasonably well on printed text
  - Module name kept as "google_vision_fallback" so app.py needs
    ZERO changes — same function signatures, same import line

SETUP (15 minutes, one-time)
──────────────────────────────
1. Install the Tesseract OCR ENGINE (not just the Python wrapper):

   Windows:
     Download installer: https://github.com/UB-Mannheim/tesseract/wiki
     Run it. During install, check "Additional language data" and
     select Hindi (this includes the Devanagari script model).
     Default install path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe

   If you forgot to select Hindi during install, download manually:
     https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata
   and place it in:
     C:\\Program Files\\Tesseract-OCR\\tessdata\\hin.traineddata

2. Install the Python wrapper:
     pip install pytesseract

3. If tesseract.exe is not on your PATH, set TESSERACT_CMD below to
   the full path (already pre-filled with the default Windows path).

That's it — no account, no key, no card needed.
"""

import os
import shutil
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    print("[Tesseract] pytesseract not installed. Run: pip install pytesseract")

VISION_ATTEMPT_THRESHOLD: float = 0.80
BLANK_THRESHOLD: float = 0.20

# Attractor classes (defined in app.py's ATTRACTOR_CLASSES) frequently
# get moderate-to-high CNN confidence on the WRONG answer — they don't
# look "uncertain" by confidence score alone. Your manuscript test
# showed avg confidence 66.6% with predictions that were still wrong,
# so a confidence-only threshold never triggers the fallback. This set
# is checked separately in should_use_fallback_for_class() below so
# Tesseract gets a chance even when the CNN looks falsely confident.
_ATTRACTOR_NAMES = {"nna", "ga", "ra", "ja", "dda", "tta", "kha"}

# ── Locate tesseract.exe ────────────────────────────────────────────
# Default Windows install path. Change this if you installed elsewhere.
_DEFAULT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def _find_tesseract() -> Optional[str]:
    # 1. Already on PATH?
    found = shutil.which("tesseract")
    if found:
        return found
    # 2. Default Windows install location?
    if os.path.exists(_DEFAULT_WIN_PATH):
        return _DEFAULT_WIN_PATH
    # 3. Environment variable override
    env_path = os.environ.get("TESSERACT_CMD")
    if env_path and os.path.exists(env_path):
        return env_path
    return None

_TESS_PATH = _find_tesseract()
if HAS_PYTESSERACT and _TESS_PATH:
    pytesseract.pytesseract.tesseract_cmd = _TESS_PATH
    print(f"[Tesseract] Using engine at: {_TESS_PATH}")
elif HAS_PYTESSERACT:
    print("[Tesseract] WARNING: tesseract.exe not found.")
    print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    print("  Or set TESSERACT_CMD environment variable to the exe path.")

HAS_TESSERACT = HAS_PYTESSERACT and _TESS_PATH is not None


# ── Devanagari character → internal class name ──────────────────────
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


def _prepare_image(img_array) -> Image.Image:
    """Convert to PIL, upscale small crops, binarize for cleaner OCR."""
    if isinstance(img_array, np.ndarray):
        pil = Image.fromarray(img_array).convert("L")  # grayscale
    else:
        pil = img_array.convert("L")

    w, h = pil.size
    if w < 100 or h < 100:
        scale = max(100 / w, 100 / h)
        # Pillow >= 9.1 exposes Image.Resampling.LANCZOS. Older versions
        # expose Image.LANCZOS directly. Use getattr chains so static
        # checkers (Pylance/Pyright) don't flag a missing attribute on
        # whichever path isn't taken, and so this works on any Pillow
        # version actually installed.
        resampling_enum = getattr(Image, "Resampling", None)
        resample = (
            getattr(resampling_enum, "LANCZOS", None) if resampling_enum is not None
            else getattr(Image, "LANCZOS", None)
        )
        if resample is None:
            resample = getattr(Image, "BICUBIC", 2)  # 2 == PIL.Image.BICUBIC value
        pil = pil.resize((int(w * scale), int(h * scale)), resample)

    return pil


def _text_to_class(text: str) -> Optional[str]:
    """Map Tesseract's Devanagari output to internal class name."""
    if not text:
        return None
    text = text.strip().replace("\n", "").replace(" ", "")
    if not text:
        return None
    if len(text) >= 2 and text[:2] in _DEVA_TO_CLASS:
        return _DEVA_TO_CLASS[text[:2]]
    if text[0] in _DEVA_TO_CLASS:
        return _DEVA_TO_CLASS[text[0]]
    return None


def google_vision_recognise(img_array) -> Tuple[Optional[str], float, str]:
    """
    Run Tesseract OCR (Devanagari/Hindi model) on a character crop.
    Same return signature as the old Vision function so app.py needs
    no changes: (class_name_or_None, confidence, source_label).
    """
    if not HAS_TESSERACT:
        return None, 0.0, "tesseract_not_installed"

    try:
        pil = _prepare_image(img_array)

        # --psm 10 = treat image as a single character
        # lang="hin" = Hindi/Devanagari trained model
        raw_text = pytesseract.image_to_string(
            pil, lang="hin", config="--psm 10"
        ).strip()

        if not raw_text:
            return None, 0.0, "tesseract_no_text"

        class_name = _text_to_class(raw_text)
        if class_name:
            return class_name, 0.70, "tesseract"

        print(f"[Tesseract] unmapped text: {repr(raw_text)}")
        return None, 0.0, "tesseract_unmapped"

    except Exception as e:
        print(f"[Tesseract] error: {e}")
        return None, 0.0, "tesseract_error"


def should_use_fallback(confidence: float) -> bool:
    return HAS_TESSERACT and confidence < VISION_ATTEMPT_THRESHOLD


def should_use_fallback_for_class(class_name: str, confidence: float) -> bool:
    """
    True if Tesseract should be tried for this prediction.
    Unlike should_use_fallback(), this also fires for known attractor
    classes even at moderate-to-high confidence, since those classes
    are wrong often enough that confidence alone is not a reliable
    signal (confirmed on your manuscript test: avg conf 66.6% but
    output was still mostly incorrect).
    """
    if not HAS_TESSERACT:
        return False
    if confidence < VISION_ATTEMPT_THRESHOLD:
        return True
    if class_name in _ATTRACTOR_NAMES and confidence < 0.95:
        return True
    return False


def is_truly_blank(confidence: float) -> bool:
    return confidence < BLANK_THRESHOLD


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC — run directly to test
# Usage: python google_vision_fallback.py [crop.png]
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  Tesseract OCR — standalone diagnostic")
    print("=" * 55)
    print(f"\npytesseract installed: {HAS_PYTESSERACT}")
    print(f"tesseract.exe found:   {_TESS_PATH or 'NOT FOUND'}")
    print(f"Ready to use:          {HAS_TESSERACT}\n")

    if not HAS_TESSERACT:
        print("Fix steps:")
        print("  1. pip install pytesseract")
        print("  2. Download/install Tesseract engine:")
        print("     https://github.com/UB-Mannheim/tesseract/wiki")
        print("  3. During install, check 'Hindi' under additional languages")
        sys.exit(1)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Testing with: {path}")
        img = np.array(Image.open(path).convert("RGB"))
        class_name, conf, source = google_vision_recognise(img)
        print(f"\nResult: class={class_name}  conf={conf:.0%}  source={source}")
    else:
        print("No image given. Pass a crop to test:")
        print("  python google_vision_fallback.py path/to/crop.png")