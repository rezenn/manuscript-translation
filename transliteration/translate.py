"""
translate.py  —  Newa Manuscript Transliteration Pipeline (v6)
Single-line mode, free Google Translate only, no paid API.
"""

import argparse, json, os, sys, shutil, tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from segment            import segment_page
from recognize          import recognize_segments
from newa_to_devanagari import predictions_to_text


def translate_free(devanagari_text: str) -> dict:
    """Free translation via Google Translate. No API key, no cost."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return {
            "success": False,
            "error": "Run: python -m pip install deep-translator",
            "english": None, "nepali": None,
        }
    clean = devanagari_text.replace("⟨?⟩", "").strip()
    if not clean:
        return {"success": False, "error": "No text after removing low-confidence characters.",
                "english": None, "nepali": None}
    try:
        english = GoogleTranslator(source="auto", target="en").translate(clean)
        try:    nepali = GoogleTranslator(source="auto", target="ne").translate(clean)
        except: nepali = None
        return {
            "success": True, "english": english, "nepali": nepali,
            "notes": (
                "Translated via Google Translate (free). "
                "Classical Nepal Bhasa may not translate perfectly. "
                "OCR errors in Devanagari will also affect translation quality."
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "english": None, "nepali": None}


def run_pipeline(
    image_path:            str,
    checkpoint_path:       str,
    translate:             bool  = False,
    keep_segments:         bool  = False,
    debug:                 bool  = False,
    confidence_threshold:  float = 0.25,
    valley_threshold:      float = None,
    min_line_height:       int   = 15,
    min_panel_gap:         int   = 30,
    min_upscale_height:    int   = 600,
    single_line:           int   = None,
) -> dict:
    result = {
        "success": False, "error": None,
        "nepal_bhasa_devanagari": None, "iast": None,
        "lines_devanagari": [], "lines_iast": [],
        "nepali": None, "english": None, "notes": None,
        "translation_error": None,
        "num_characters": 0, "num_lines": 0,
        "avg_confidence": 0.0, "low_conf_count": 0,
        "flagged_chars": [], "segments_dir": None,
        "single_line_mode": single_line is not None,
        "single_line_index": single_line,
    }

    segments_dir = (
        str(Path("output_segments") / Path(image_path).stem)
        if keep_segments else tempfile.mkdtemp(prefix="newa_seg_")
    )

    try:
        print("\n[1/4] Segmenting manuscript page...")
        metadata = segment_page(
            image_path         = image_path,
            output_dir         = segments_dir,
            debug              = debug,
            valley_threshold   = valley_threshold,
            min_line_height    = min_line_height,
            min_panel_gap      = min_panel_gap,
            min_upscale_height = min_upscale_height,
        )
        if not metadata:
            result["error"] = (
                "No characters detected. Use --debug to inspect. "
                "Try --seg-threshold 0.15 if lines are merged."
            )
            return result

        print(f"\n[2/4] Running OCR on {len(metadata)} characters...")
        predictions = recognize_segments(
            segments_dir         = segments_dir,
            checkpoint_path      = checkpoint_path,
            confidence_threshold = confidence_threshold,
        )
        if not predictions:
            result["error"] = "OCR returned no predictions. Check --checkpoint path."
            return result

        print("\n[3/4] Converting to Devanagari...")
        deva_all = predictions_to_text(predictions, output_format="devanagari")
        iast_all = predictions_to_text(predictions, output_format="iast")
        confs = [p.get("confidence", 1.0) for p in predictions]

        result.update({
            "lines_devanagari": deva_all["lines"],
            "lines_iast":       iast_all["lines"],
            "num_characters":   len(predictions),
            "num_lines":        len(deva_all["lines"]),
            "avg_confidence":   round(sum(confs) / len(confs), 4) if confs else 0,
            "low_conf_count":   sum(1 for p in predictions if p.get("low_conf")),
            "flagged_chars":    deva_all["flagged"],
        })

        if single_line is not None:
            n_lines = len(deva_all["lines"])
            if single_line >= n_lines:
                result["error"] = (
                    f"Line {single_line + 1} requested but only {n_lines} lines found. "
                    f"Valid range: 1 to {n_lines}."
                )
                return result
            deva_text = deva_all["lines"][single_line]
            iast_text = iast_all["lines"][single_line]
            line_preds = [p for p in predictions if p.get("line") == single_line]
            line_confs = [p.get("confidence", 1.0) for p in line_preds]
            result.update({
                "num_characters": len(line_preds),
                "avg_confidence": round(sum(line_confs)/len(line_confs), 4) if line_confs else 0,
                "low_conf_count": sum(1 for p in line_preds if p.get("low_conf")),
            })
        else:
            deva_text = deva_all["text"]
            iast_text = iast_all["text"]

        result["nepal_bhasa_devanagari"] = deva_text
        result["iast"] = iast_text

        if translate:
            print("\n[4/4] Translating with Google Translate (free)...")
            trans = translate_free(deva_text)
            if trans.get("success"):
                result.update({
                    "nepali": trans.get("nepali"),
                    "english": trans.get("english"),
                    "notes": trans.get("notes"),
                })
            else:
                result["translation_error"] = trans.get("error")
                print(f"  Translation error: {result['translation_error']}")
        else:
            print("\n[4/4] Translation skipped (use --translate to enable free Google Translate)")
            result["translation_error"] = "Translation skipped. Use --translate to enable."

        result["success"] = True
        if keep_segments:
            result["segments_dir"] = segments_dir

    except Exception as e:
        import traceback
        result["error"] = f"Pipeline error: {e}"
        print(f"\nERROR: {e}\n{traceback.format_exc()}")
    finally:
        if not keep_segments and segments_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(segments_dir, ignore_errors=True)

    return result


def save_report(result: dict, image_path: str, output_prefix: str = None):
    stem = Path(image_path).stem
    if result.get("single_line_mode") and result.get("single_line_index") is not None:
        stem += f"_line{result['single_line_index'] + 1}"
    base = Path(output_prefix or f"transliteration_output/{stem}")
    base.parent.mkdir(parents=True, exist_ok=True)
    sep = "=" * 65; sub = "-" * 65

    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"{sep}\nNEWA MANUSCRIPT TRANSLITERATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Source:    {image_path}\n")
        if result.get("single_line_mode"):
            f.write(f"Mode:      Single line (line {result['single_line_index'] + 1})\n")
        f.write(f"{sep}\n\n")
        f.write(f"Characters: {result['num_characters']}  Lines: {result['num_lines']}  "
                f"Avg conf: {result['avg_confidence']:.1%}  Low-conf: {result['low_conf_count']}\n\n")
        if result.get("lines_devanagari") and not result.get("single_line_mode"):
            f.write(f"{sub}\nPER-LINE DEVANAGARI\n{sub}\n")
            for i, line in enumerate(result["lines_devanagari"]):
                f.write(f"  Line {i+1:02d}: {line}\n")
            f.write("\n")
        f.write(f"{sub}\nNEPAL BHASA (Devanagari)\n{sub}\n{result.get('nepal_bhasa_devanagari','N/A')}\n\n")
        f.write(f"{sub}\nIAST Romanization\n{sub}\n{result.get('iast','N/A')}\n\n")
        for label, key in [("Nepali Translation","nepali"),("English Translation","english"),("Notes","notes")]:
            if result.get(key):
                f.write(f"{sub}\n{label}\n{sub}\n{result[key]}\n\n")
        if result.get("translation_error"):
            f.write(f"Translation: {result['translation_error']}\n")

    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n  Report -> {base.with_suffix('.txt')}")
    print(f"  JSON   -> {base.with_suffix('.json')}")
    return str(base.with_suffix(".txt")), str(base.with_suffix(".json"))


def parse_args():
    p = argparse.ArgumentParser(
        description="Newa transliteration pipeline v6",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full manuscript:
  python transliteration/translate.py --image manuscript.jpg --checkpoint checkpoints/best_model.pth

  # With free translation:
  python transliteration/translate.py --image manuscript.jpg --checkpoint checkpoints/best_model.pth --translate

  # Single line only (e.g. line 3):
  python transliteration/translate.py --image manuscript.jpg --checkpoint checkpoints/best_model.pth --line 3 --translate

  # Debug segmentation:
  python transliteration/translate.py --image manuscript.jpg --checkpoint checkpoints/best_model.pth --keep-segments --debug
"""
    )
    p.add_argument("--image",               required=True)
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--translate",           action="store_true",
                   help="Enable free Google Translate (no API key needed)")
    p.add_argument("--line",                type=int, default=None,
                   help="Read only this line number (1-based, e.g. --line 3)")
    p.add_argument("--keep-segments",       action="store_true")
    p.add_argument("--debug",               action="store_true")
    p.add_argument("--output",              default=None)
    p.add_argument("--confidence",          type=float, default=0.25)
    p.add_argument("--seg-threshold",       type=float, default=None,
                   help="Valley threshold (auto by default). Try 0.15 if lines merge.")
    p.add_argument("--seg-min-line-height", type=int, default=15)
    p.add_argument("--seg-min-panel-gap",   type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    single_line = (args.line - 1) if args.line is not None else None

    result = run_pipeline(
        image_path           = args.image,
        checkpoint_path      = args.checkpoint,
        translate            = args.translate,
        keep_segments        = args.keep_segments,
        debug                = args.debug,
        confidence_threshold = args.confidence,
        valley_threshold     = args.seg_threshold,
        min_line_height      = args.seg_min_line_height,
        min_panel_gap        = args.seg_min_panel_gap,
        single_line          = single_line,
    )

    if not result["success"]:
        print(f"\nERROR: {result['error']}")
        sys.exit(1)

    sep = "=" * 65
    print(f"\n{sep}\nRESULTS\n{sep}")
    if result.get("single_line_mode"):
        print(f"Mode: Single line (line {args.line})")
    print(f"Characters: {result['num_characters']}  Lines: {result['num_lines']}  "
          f"Avg conf: {result['avg_confidence']:.1%}")

    if result.get("lines_devanagari") and not result.get("single_line_mode"):
        print("\n-- Per-line Devanagari --")
        for i, line in enumerate(result["lines_devanagari"]):
            print(f"  Line {i+1:02d}: {line}")

    print(f"\n-- Devanagari --\n{result['nepal_bhasa_devanagari']}")
    print(f"\n-- IAST --\n{result['iast']}")
    for label, key in [("Nepali", "nepali"), ("English", "english"), ("Notes", "notes")]:
        if result.get(key):
            print(f"\n-- {label} --\n{result[key]}")
    if result.get("translation_error"):
        print(f"\n-- Translation --\n{result['translation_error']}")

    save_report(result, args.image, args.output)