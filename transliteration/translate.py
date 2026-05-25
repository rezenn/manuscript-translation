"""
translate.py  —  End-to-End Newa Manuscript Transliteration
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
Full pipeline: manuscript photo → 3 outputs:
  1. Nepal Bhasa in Devanagari  (your OCR output — the script rendered)
  2. Modern Nepali translation   (Claude API)
  3. English translation         (Claude API)

WHY 3 OUTPUTS?
──────────────
  - Newa script  = the writing system (like how English uses Latin letters)
  - Nepal Bhasa  = the language in those manuscripts
  - Devanagari   = a different writing system that can write the same language

  Step 1 (OCR):    Newa script → Nepal Bhasa written in Devanagari
                   (same language, different alphabet — like transliteration)
  Step 2 (Claude): Nepal Bhasa → modern Nepali  (actual translation)
  Step 3 (Claude): modern Nepali → English       (another translation)

Run from command line:
    python transliteration/translate.py \\
        --image manuscript.jpg \\
        --checkpoint checkpoints/best_model.pth

    # All 3 outputs with Claude translation:
    python transliteration/translate.py \\
        --image manuscript.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --translate

Or import and call run_pipeline() from app.py (the Gradio UI).
"""

import argparse
import json
import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from segment import segment_page
from recognize import recognize_segments
from newa_to_devanagari import predictions_to_text


# ═══════════════════════════════════════════════════════════════════
# CLAUDE API: TRANSLATE NEPAL BHASA → NEPALI + ENGLISH
# ═══════════════════════════════════════════════════════════════════

def translate_with_claude(devanagari_text: str, iast_text: str) -> dict:
    """
    Send Nepal Bhasa (written in Devanagari) to Claude API.
    Returns Nepali translation and English translation.

    WHAT CLAUDE IS DOING HERE:
    ─────────────────────────
    Your OCR converted Newa script → Devanagari letters.
    The result is classical Nepal Bhasa — the language of Kathmandu Valley
    manuscripts, related to but different from modern Nepali.

    Claude acts as a bilingual scholar who:
      1. Reads the classical Nepal Bhasa text
      2. Writes the same meaning in modern Nepali (same family, easier to read)
      3. Translates to English for international readers
      4. Explains any difficult vocabulary or historical context

    REQUIRES: ANTHROPIC_API_KEY environment variable
    Get one at: https://console.anthropic.com
    """
    try:
        import anthropic
    except ImportError:
        return {
            "success": False,
            "error": "anthropic package not installed.\nRun: pip install anthropic",
            "nepali": None,
            "english": None,
            "notes": None,
            "confidence": None,
        }

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": (
                "ANTHROPIC_API_KEY not set.\n"
                "1. Go to https://console.anthropic.com\n"
                "2. Create an API key\n"
                "3. Run: export ANTHROPIC_API_KEY='sk-ant-...'\n"
                "   Or enter it in the UI settings."
            ),
            "nepali": None,
            "english": None,
            "notes": None,
            "confidence": None,
        }

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an expert scholar in classical Nepal Bhasa (Newa language) from the Kathmandu Valley. You specialise in historical manuscripts written in Newa/Prachalit script, now transliterated into Devanagari.

I have used an OCR system to read a Newa manuscript. The OCR has converted the visual Newa script into Devanagari letters. The result is classical Nepal Bhasa — the language of the Kathmandu Valley, written in Devanagari instead of the original Newa script.

Note: Characters marked ⟨?⟩ were low-confidence in the OCR and may be incorrect.

IAST transliteration (phonetic, for reference):
{iast_text}

Nepal Bhasa in Devanagari (OCR output):
{devanagari_text}

Please provide:
1. A modern Nepali translation (same script, more accessible to modern readers)
2. An English translation
3. Brief scholarly notes: historical context, unusual vocabulary, possible OCR errors, what type of text this appears to be (religious, historical, legal, literary, etc.)
4. Your confidence level (high/medium/low) based on text clarity and your certainty

Respond ONLY in this exact JSON format, no preamble, no markdown fences:
{{
  "nepali": "आधुनिक नेपाली अनुवाद यहाँ",
  "english": "English translation here",
  "notes": "Scholarly notes about the text, vocabulary, historical context, and any OCR corrections",
  "confidence": "high/medium/low",
  "text_type": "religious/historical/legal/literary/administrative/unknown"
}}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = message.content[0].text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.split("\n")
                if not line.startswith("```")
            ).strip()

        result = json.loads(raw)
        result["success"] = True
        return result

    except json.JSONDecodeError:
        return {
            "success": True,
            "nepali":  None,
            "english": message.content[0].text,
            "notes":   "Could not parse structured response — showing raw output.",
            "confidence": "unknown",
            "text_type": "unknown",
        }
    except Exception as e:
        return {
            "success": False,
            "error":   str(e),
            "nepali":  None,
            "english": None,
            "notes":   None,
            "confidence": None,
        }


# ═══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════

def run_pipeline(
    image_path: str,
    checkpoint_path: str,
    api_key: str         = None,   # can be passed directly from UI
    keep_segments: bool  = False,
    debug: bool          = False,
    confidence_threshold: float = 0.3,
) -> dict:
    """
    Run the full transliteration pipeline.

    Returns a dict with ALL results:
    {
        "success": True/False,
        "error":   error message if failed,

        # OCR results
        "nepal_bhasa_devanagari": "क...",   ← Nepal Bhasa in Devanagari letters
        "iast":                   "ka...",  ← IAST romanization
        "num_characters":         123,
        "num_lines":              8,
        "avg_confidence":         0.94,
        "low_conf_count":         3,
        "flagged_chars":          [...],

        # Translation results (if API key provided)
        "nepali":       "आधुनिक नेपाली...",
        "english":      "English translation...",
        "notes":        "Scholarly notes...",
        "text_type":    "religious",
        "translation_confidence": "high",

        # Segments dir (if keep_segments=True)
        "segments_dir": "output_segments/",
    }
    """
    # Allow API key to be passed directly from UI (overrides env var)
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    result = {
        "success": False,
        "error": None,
        "nepal_bhasa_devanagari": None,
        "iast": None,
        "nepali": None,
        "english": None,
        "notes": None,
        "text_type": None,
        "translation_confidence": None,
        "num_characters": 0,
        "num_lines": 0,
        "avg_confidence": 0.0,
        "low_conf_count": 0,
        "flagged_chars": [],
        "segments_dir": None,
    }

    # Set up segments directory
    if keep_segments:
        stem = Path(image_path).stem
        segments_dir = f"output_segments/{stem}"
    else:
        segments_dir = tempfile.mkdtemp(prefix="newa_seg_")

    try:
        # ── STEP 1: Segment ───────────────────────────────────────
        print("\n[1/3] Segmenting manuscript page...")
        metadata = segment_page(
            image_path=image_path,
            output_dir=segments_dir,
            debug=debug,
        )

        if not metadata:
            result["error"] = (
                "No characters detected in the image.\n\n"
                "Tips:\n"
                "• Make sure the image shows clear dark ink on a light background\n"
                "• Try a higher-resolution photo\n"
                "• Avoid heavy shadows or glare\n"
                "• Run with --debug to see what the segmenter detected"
            )
            return result

        # ── STEP 2: Recognize ─────────────────────────────────────
        print(f"\n[2/3] Running OCR on {len(metadata)} characters...")
        predictions = recognize_segments(
            segments_dir=segments_dir,
            checkpoint_path=checkpoint_path,
            confidence_threshold=confidence_threshold,
        )

        if not predictions:
            result["error"] = "OCR produced no predictions. Check the checkpoint path."
            return result

        # ── STEP 3: Convert to Devanagari ─────────────────────────
        print(f"\n[3/3] Converting to Devanagari...")
        deva_result = predictions_to_text(predictions, output_format="devanagari")
        iast_result = predictions_to_text(predictions, output_format="iast")

        # Stats
        confs = [p.get("confidence", 1.0) for p in predictions]
        avg_conf = sum(confs) / len(confs) if confs else 0
        low_conf_count = sum(1 for p in predictions if p.get("low_conf", False))

        result.update({
            "nepal_bhasa_devanagari": deva_result["text"],
            "iast":                   iast_result["text"],
            "num_characters":         len(predictions),
            "num_lines":              len(deva_result["lines"]),
            "avg_confidence":         round(avg_conf, 4),
            "low_conf_count":         low_conf_count,
            "flagged_chars":          deva_result["flagged"],
        })

        # ── STEP 4: Claude Translation ────────────────────────────
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("\n[4/4] Translating with Claude API...")
            translation = translate_with_claude(
                devanagari_text=deva_result["text"],
                iast_text=iast_result["text"],
            )
            if translation.get("success"):
                result.update({
                    "nepali":                 translation.get("nepali"),
                    "english":                translation.get("english"),
                    "notes":                  translation.get("notes"),
                    "text_type":              translation.get("text_type"),
                    "translation_confidence": translation.get("confidence"),
                })
            else:
                result["translation_error"] = translation.get("error")
        else:
            print("\n[4/4] Skipping translation (no API key set)")
            result["translation_error"] = (
                "No ANTHROPIC_API_KEY set. Enter your API key in the UI to enable translation."
            )

        result["success"] = True
        if keep_segments:
            result["segments_dir"] = segments_dir

    except Exception as e:
        result["error"] = f"Pipeline error: {str(e)}"
        import traceback
        result["traceback"] = traceback.format_exc()

    finally:
        # Clean up temp segments
        if not keep_segments and segments_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(segments_dir, ignore_errors=True)

    return result


# ═══════════════════════════════════════════════════════════════════
# SAVE REPORT
# ═══════════════════════════════════════════════════════════════════

def save_report(result: dict, image_path: str, output_prefix: str = None):
    """Save a .txt and .json report from the pipeline result."""
    stem = Path(image_path).stem
    out  = Path(output_prefix or f"transliteration_output/{stem}")
    out.parent.mkdir(parents=True, exist_ok=True)

    txt_path = out.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("═" * 65 + "\n")
        f.write("NEWA MANUSCRIPT TRANSLITERATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Source:    {image_path}\n")
        f.write("═" * 65 + "\n\n")

        f.write(f"Characters: {result['num_characters']}\n")
        f.write(f"Lines:      {result['num_lines']}\n")
        f.write(f"Avg OCR confidence: {result['avg_confidence']:.1%}\n")
        f.write(f"Low-conf chars:     {result['low_conf_count']}\n\n")

        f.write("─" * 65 + "\n")
        f.write("NEPAL BHASA (Devanagari transcription from OCR)\n")
        f.write("─" * 65 + "\n")
        f.write((result["nepal_bhasa_devanagari"] or "N/A") + "\n\n")

        f.write("─" * 65 + "\n")
        f.write("IAST ROMANIZATION\n")
        f.write("─" * 65 + "\n")
        f.write((result["iast"] or "N/A") + "\n\n")

        if result.get("nepali"):
            f.write("─" * 65 + "\n")
            f.write("MODERN NEPALI TRANSLATION\n")
            f.write("─" * 65 + "\n")
            f.write(result["nepali"] + "\n\n")

        if result.get("english"):
            f.write("─" * 65 + "\n")
            f.write("ENGLISH TRANSLATION\n")
            f.write("─" * 65 + "\n")
            f.write(result["english"] + "\n\n")

        if result.get("notes"):
            f.write("─" * 65 + "\n")
            f.write("SCHOLARLY NOTES\n")
            f.write("─" * 65 + "\n")
            f.write(result["notes"] + "\n\n")

        if result.get("flagged_chars"):
            f.write("─" * 65 + "\n")
            f.write("LOW-CONFIDENCE CHARACTERS\n")
            f.write("─" * 65 + "\n")
            for item in result["flagged_chars"]:
                f.write(f"  {item['file']}  →  {item['predicted']}  "
                        f"({item['confidence']:.1%})\n")

    json_path = out.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Report → {txt_path}")
    print(f"  JSON   → {json_path}")
    return str(txt_path), str(json_path)


# ═══════════════════════════════════════════════════════════════════
# COMMAND LINE
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Newa manuscript transliteration pipeline")
    p.add_argument("--image",       required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--translate",   action="store_true",
                   help="Use Claude API to translate to Nepali + English")
    p.add_argument("--api-key",     default=None,
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    p.add_argument("--keep-segments", action="store_true")
    p.add_argument("--debug",       action="store_true")
    p.add_argument("--output",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.translate and args.api_key:
        os.environ["ANTHROPIC_API_KEY"] = args.api_key

    if not args.translate:
        # Don't use API even if key is set
        key_backup = os.environ.pop("ANTHROPIC_API_KEY", None)

    result = run_pipeline(
        image_path      = args.image,
        checkpoint_path = args.checkpoint,
        keep_segments   = args.keep_segments,
        debug           = args.debug,
    )

    if not result["success"]:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    print(f"\n{'═'*65}")
    print("RESULTS")
    print(f"{'═'*65}")
    print(f"\nNepal Bhasa (Devanagari):\n{result['nepal_bhasa_devanagari']}")
    if result.get("nepali"):
        print(f"\nNepali:\n{result['nepali']}")
    if result.get("english"):
        print(f"\nEnglish:\n{result['english']}")

    save_report(result, args.image, args.output)