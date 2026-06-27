"""
translate.py  —  Newa End-to-End Transliteration Runner (v7)
═══════════════════════════════════════════════════════════════════

CHANGES vs v6
─────────────
• Line indexing fix: --line N now uses 1-based indexing consistently
  (line 1 = first line visible in the manuscript).
• Better error message when requested line doesn't exist.
• Devanagari conversion uses 4-step alias lookup (handles uppercase
  vowel class names like vowel_A, vowel_AA).
• --single-char mode: route a single character image directly to
  recognize_single() bypassing segmentation entirely.
• Removed accidental seaborn dependency.
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# ── path setup ─────────────────────────────────────────────────────
HERE = Path(__file__).parent
ROOT = HERE.parent if HERE.name == "transliteration" else HERE
sys.path.insert(0, str(ROOT / "ocr_model"))
sys.path.insert(0, str(ROOT / "transliteration"))

from segment   import segment_page
from recognize import recognize_segments, recognize_single

try:
    from newa_to_devanagari import to_devanagari, to_iast
    HAS_DEVA_MODULE = True
except ImportError:
    to_devanagari = None
    to_iast = None
    HAS_DEVA_MODULE = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATE = True
except ImportError:
    GoogleTranslator = None
    HAS_TRANSLATE = False


# ══════════════════════════════════════════════════════════════════
# INLINE FALLBACK CHAR MAP (used if newa_to_devanagari.py missing)
# ══════════════════════════════════════════════════════════════════

DEVA_MAP = {
    "ka":"क","kha":"ख","ga":"ग","gha":"घ","nga":"ङ",
    "ca":"च","cha":"छ","ja":"ज","jha":"झ","nya":"ञ",
    "tta":"ट","ttha":"ठ","dda":"ड","ddha":"ढ","nna":"ण",
    "ta":"त","tha":"थ","da":"द","dha":"ध","na":"न",
    "pa":"प","pha":"फ","ba":"ब","bha":"भ","ma":"म",
    "ya":"य","ra":"र","la":"ल","wa":"व","sa":"स",
    "sha":"श","ssa":"ष","ha":"ह",
    "vowel_a":"अ","vowel_aa":"आ","vowel_i":"इ","vowel_ii":"ई",
    "vowel_u":"उ","vowel_uu":"ऊ","vowel_e":"ए","vowel_ai":"ऐ",
    "vowel_o":"ओ","vowel_au":"औ",
    "matra_aa":"ा","matra_i":"ि","matra_ii":"ी",
    "matra_u":"ु","matra_uu":"ू","matra_e":"े","matra_ai":"ै",
    "matra_o":"ो","matra_au":"ौ",
    "anusvara":"ं","visarga":"ः","candrabindu":"ँ",
    "virama":"्","avagraha":"ऽ",
    "digit_0":"०","digit_1":"१","digit_2":"२","digit_3":"३",
    "digit_4":"४","digit_5":"५","digit_6":"६","digit_7":"७",
    "digit_8":"८","digit_9":"९",
}

IAST_MAP = {
    "ka":"k","kha":"kh","ga":"g","gha":"gh","nga":"ṅ",
    "ca":"c","cha":"ch","ja":"j","jha":"jh","nya":"ñ",
    "tta":"ṭ","ttha":"ṭh","dda":"ḍ","ddha":"ḍh","nna":"ṇ",
    "ta":"t","tha":"th","da":"d","dha":"dh","na":"n",
    "pa":"p","pha":"ph","ba":"b","bha":"bh","ma":"m",
    "ya":"y","ra":"r","la":"l","wa":"v","sa":"s",
    "sha":"ś","ssa":"ṣ","ha":"h",
    "vowel_a":"a","vowel_aa":"ā","vowel_i":"i","vowel_ii":"ī",
    "vowel_u":"u","vowel_uu":"ū","vowel_e":"e","vowel_ai":"ai",
    "vowel_o":"o","vowel_au":"au",
    "matra_aa":"ā","matra_i":"i","matra_ii":"ī",
    "matra_u":"u","matra_uu":"ū","matra_e":"e","matra_ai":"ai",
    "matra_o":"o","matra_au":"au",
    "anusvara":"ṃ","visarga":"ḥ","candrabindu":"m̐",
    "virama":"·","avagraha":"ʼ",
    "digit_0":"0","digit_1":"1","digit_2":"2","digit_3":"3",
    "digit_4":"4","digit_5":"5","digit_6":"6","digit_7":"7",
    "digit_8":"8","digit_9":"9",
}


def char_to_deva(name: str) -> str:
    """4-step lookup: module → as-is → lowercase → stripped."""
    if HAS_DEVA_MODULE and to_devanagari is not None:
        try:
            return to_devanagari(name)
        except Exception:
            pass
    return (DEVA_MAP.get(name)
            or DEVA_MAP.get(name.lower())
            or DEVA_MAP.get(name.lower().replace("vowel_", "vowel_"))
            or "⟨?⟩")


def char_to_iast(name: str) -> str:
    if HAS_DEVA_MODULE and to_iast is not None:
        try:
            return to_iast(name)
        except Exception:
            pass
    return (IAST_MAP.get(name)
            or IAST_MAP.get(name.lower())
            or "?")


# ══════════════════════════════════════════════════════════════════
# SINGLE CHARACTER MODE  (bypasses segmentation)
# ══════════════════════════════════════════════════════════════════

def run_single_char(args):
    """Direct inference on one character image."""
    print("\n[Single-character mode — no segmentation]\n")
    result = recognize_single(args.image, args.checkpoint)
    if result is None:
        sys.exit(1)
    best = result["predicted"]
    conf = result["confidence"]
    print(f"\nDevanagari: {char_to_deva(best)}")
    print(f"IAST:       {char_to_iast(best)}")
    return result


# ══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_full(args):
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        sys.exit(1)

    output_dir = Path(args.out) if args.out else Path("transliteration_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix="newa_seg_")

    try:
        # ── 1. Segment ─────────────────────────────────────────────
        print("\n[1/4] Segmenting manuscript page...\n")
        seg_dir = os.path.join(tmp_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        segment_page(
            image_path=str(image_path),
            output_dir=seg_dir,
            valley_threshold=args.seg_threshold,
        )

        # ── 2. OCR ─────────────────────────────────────────────────
        meta_path = Path(seg_dir) / "segments_meta.json"
        if not meta_path.exists():
            print("ERROR: segmentation produced no metadata. Check segment.py output above.")
            sys.exit(1)

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        total_chars = meta.get("num_chars", 0)
        print(f"\n[2/4] Running OCR on {total_chars} characters...\n")

        char_list = recognize_segments(
            segments_dir=seg_dir,
            checkpoint_path=args.checkpoint,
            confidence_threshold=args.conf_threshold,
        )

        # ── 3. Group by line (1-based for user display) ─────────────
        lines = {}
        for c in sorted(char_list, key=lambda x: (x["line"], x["char_idx"])):
            lines.setdefault(c["line"], []).append(c)

        num_lines = len(lines)
        sorted_line_keys = sorted(lines.keys())

        print(f"\n[3/4] Converting to Devanagari...\n")

        # Build per-line Devanagari
        line_deva = {}
        for ln_key in sorted_line_keys:
            chars = lines[ln_key]
            s = ""
            for c in chars:
                pred = c.get("predicted", "")
                conf = c.get("confidence", 0.0)
                s += char_to_deva(pred) if not c.get("low_conf", conf < args.conf_threshold) else "⟨?⟩"
            line_deva[ln_key] = s

        # Determine which lines to output
        if args.line is not None:
            # User requested a specific line (1-based)
            requested = args.line - 1   # convert to 0-based internal key
            if requested not in lines:
                valid_range = f"1 to {num_lines}"
                print(f"ERROR: Line {args.line} requested but only {num_lines} line(s) found. "
                      f"Valid range: {valid_range}.")
                sys.exit(1)
            output_line_keys = [requested]
            mode_label = f"Single line (line {args.line})"
        else:
            output_line_keys = sorted_line_keys
            mode_label = "All lines"

        # ── 4. Translate ────────────────────────────────────────────
        concat_for_translate = "".join(
            char_to_deva(c.get("predicted", ""))
            for lk in output_line_keys
            for c in lines[lk]
            if not c.get("low_conf", c.get("confidence", 0) < args.conf_threshold)
        )

        translation = ""
        if args.translate:
            print("\n[4/4] Translating via Google Translate (free)...\n")
            if HAS_TRANSLATE and GoogleTranslator is not None:
                try:
                    clean = concat_for_translate.replace("⟨?⟩", "").strip()
                    if clean:
                        translation = GoogleTranslator(source="ne", target="en").translate(clean)
                    else:
                        translation = "(nothing to translate — all characters low-confidence)"
                except Exception as e:
                    translation = f"(translation error: {e})"
                print(f"  Translation: {translation}")
            else:
                translation = "(deep-translator not installed — run: pip install deep-translator)"
                print(f"  {translation}")

        # ── Print results ───────────────────────────────────────────
        total_chars_out = sum(len(lines[lk]) for lk in output_line_keys)
        avg_conf = (
            sum(c.get("confidence", 0) for lk in output_line_keys for c in lines[lk])
            / total_chars_out if total_chars_out else 0
        )

        print("\n" + "=" * 65)
        print("RESULTS")
        print("=" * 65)
        print(f"Mode: {mode_label}")
        print(f"Characters: {total_chars_out}  Lines: {num_lines}  Avg conf: {avg_conf:.1%}")
        print()
        print("-- Per-line Devanagari --")
        for lk in sorted_line_keys:
            marker = " ◄" if lk in output_line_keys and args.line else ""
            print(f"  Line {lk+1:02d}: {line_deva[lk]}{marker}")

        output_deva = "".join(line_deva[lk] for lk in output_line_keys)
        output_iast = "".join(
            char_to_iast(c.get("predicted", ""))
            for lk in output_line_keys
            for c in lines[lk]
            if not c.get("low_conf", c.get("confidence", 0) < args.conf_threshold)
        )

        print(f"\n-- Devanagari --\n{output_deva}")
        print(f"\n-- IAST --\n{output_iast}")
        if translation:
            print(f"\n-- English --\n{translation}")

        # ── Save report ─────────────────────────────────────────────
        stem  = image_path.stem
        label = f"_line{args.line}" if args.line else ""
        txt_path  = output_dir / f"{stem}{label}.txt"
        json_path = output_dir / f"{stem}{label}.json"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {image_path}\n")
            f.write(f"Date:   {datetime.now().isoformat()}\n")
            f.write(f"Mode:   {mode_label}\n\n")
            for lk in sorted_line_keys:
                f.write(f"Line {lk+1:02d}: {line_deva[lk]}\n")
            f.write(f"\nDevanagari:\n{output_deva}\n")
            f.write(f"\nIAST:\n{output_iast}\n")
            if translation:
                f.write(f"\nEnglish:\n{translation}\n")

        report = {
            "source": str(image_path),
            "date":   datetime.now().isoformat(),
            "mode":   mode_label,
            "num_lines": num_lines,
            "per_line_devanagari": {str(lk+1): line_deva[lk] for lk in sorted_line_keys},
            "devanagari": output_deva,
            "iast":       output_iast,
            "translation": translation,
            "characters": char_list,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n  Report -> {txt_path}")
        print(f"  JSON   -> {json_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Newa manuscript transliterator v7")
    p.add_argument("--image",         required=True, help="Input image path")
    p.add_argument("--checkpoint",    required=True, help="Model checkpoint (.pth)")
    p.add_argument("--out",           default=None,  help="Output directory")
    p.add_argument("--line",          type=int, default=None,
                   help="Output only this line (1-based). Omit for all lines.")
    p.add_argument("--translate",     action="store_true",
                   help="Translate Devanagari output to English")
    p.add_argument("--single-char",   action="store_true",
                   help="Treat --image as a single character (bypass segmentation)")
    p.add_argument("--conf-threshold", type=float, default=0.55,
                   help="Min confidence (and margin) to accept a character "
                        "prediction; below this, output ⟨?⟩ instead of a "
                        "guess. Raised from 0.25 → 0.55 default (see "
                        "recognize.py docstring for why).")
    p.add_argument("--seg-threshold", type=float, default=None,
                   help="Segmentation sensitivity (0.05–0.20, lower = more lines)")
    p.add_argument("--keep-segments", action="store_true",
                   help="Don't delete temp segment files (for debugging)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.single_char:
        run_single_char(args)
    else:
        run_full(args)