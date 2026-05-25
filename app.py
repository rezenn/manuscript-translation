"""
app.py  —  Newa Manuscript Transliteration Web UI
═══════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
───────────────────
Launches a browser-based web application for your Newa OCR thesis.
Upload a manuscript photo → get results in all 3 languages.

HOW TO INSTALL GRADIO
─────────────────────
    pip install gradio

HOW TO RUN
──────────
    # From your project root directory:
    python app.py

    # This will print something like:
    #   Running on local URL:  http://127.0.0.1:7860
    #   Running on public URL: https://abc123.gradio.live  (shareable link!)

    # Open the local URL in your browser.
    # Share the public URL with your thesis supervisor.

WHAT THE UI SHOWS
─────────────────
  Left panel:
    - Image upload
    - API key input (for translation)
    - Settings (checkpoint path, confidence threshold)
    - Run button

  Right panel (results):
    - Tab 1: Nepal Bhasa in Devanagari  (your OCR output)
    - Tab 2: Modern Nepali              (Claude translation)
    - Tab 3: English                    (Claude translation)
    - Tab 4: Scholarly Notes + Stats
    - Tab 5: Flagged characters (low-confidence OCR)

PROJECT STRUCTURE EXPECTED
──────────────────────────
    your_project/
    ├── app.py                      ← this file (run from here)
    ├── checkpoints/
    │   └── best_model.pth          ← your trained model
    ├── transliteration/
    │   ├── segment.py
    │   ├── recognize.py
    │   ├── newa_to_devanagari.py
    │   └── translate.py
    └── ocr_model/
        └── model.py
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

import gradio as gr

# Add paths so imports work
sys.path.insert(0, str(Path(__file__).parent / "transliteration"))
sys.path.insert(0, str(Path(__file__).parent / "ocr_model"))

from translate import run_pipeline, save_report


# ═══════════════════════════════════════════════════════════════════
# DEFAULT CHECKPOINT PATH
# ═══════════════════════════════════════════════════════════════════

DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"


# ═══════════════════════════════════════════════════════════════════
# CORE FUNCTION (called when user clicks "Transliterate")
# ═══════════════════════════════════════════════════════════════════

def transliterate(
    image,
    checkpoint_path,
    api_key,
    confidence_threshold,
    keep_segments,
    debug_mode,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Called by Gradio when the user clicks the Transliterate button.

    Parameters come from the UI widgets.
    Returns values that fill the output widgets.
    """

    # ── Validate inputs ───────────────────────────────────────────
    if image is None:
        return (
            "⚠️ Please upload a manuscript image first.",
            "", "", "", "", "", build_stats_html(None)
        )

    if not checkpoint_path or not Path(checkpoint_path).exists():
        return (
            f"⚠️ Model checkpoint not found: {checkpoint_path}\n\n"
            "Make sure you have trained the model and the path is correct.\n"
            "Default location: checkpoints/best_model.pth",
            "", "", "", "", "", build_stats_html(None)
        )

    # Save uploaded image to a temp file
    # (Gradio gives us a PIL Image or numpy array; we need a file path)
    import numpy as np
    import cv2

    tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    if isinstance(image, np.ndarray):
        cv2.imwrite(tmp_img.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        image.save(tmp_img.name)
    tmp_img.close()

    progress(0.1, desc="Starting pipeline...")

    # ── Run the full pipeline ─────────────────────────────────────
    try:
        result = run_pipeline(
            image_path           = tmp_img.name,
            checkpoint_path      = checkpoint_path,
            api_key              = api_key.strip() if api_key else None,
            keep_segments        = keep_segments,
            debug                = debug_mode,
            confidence_threshold = float(confidence_threshold),
        )
    except Exception as e:
        import traceback
        return (
            f"❌ Pipeline failed:\n{str(e)}\n\n{traceback.format_exc()}",
            "", "", "", "", "", build_stats_html(None)
        )
    finally:
        os.unlink(tmp_img.name)  # clean up temp image

    progress(0.9, desc="Formatting results...")

    # ── Handle errors ─────────────────────────────────────────────
    if not result["success"]:
        return (
            f"❌ Error:\n{result['error']}",
            "", "", "", "", "", build_stats_html(None)
        )

    # ── Format outputs ────────────────────────────────────────────

    # Tab 1: Nepal Bhasa (Devanagari)
    nepal_bhasa = result["nepal_bhasa_devanagari"] or ""
    if result["low_conf_count"] > 0:
        nepal_bhasa += (
            f"\n\n---\n⚠️ {result['low_conf_count']} characters marked ⟨?⟩ "
            f"had low OCR confidence and may be incorrect."
        )

    # Tab 2: Nepali
    if result.get("nepali"):
        nepali = result["nepali"]
    elif result.get("translation_error"):
        nepali = f"Translation not available:\n{result['translation_error']}"
    else:
        nepali = "Translation not performed."

    # Tab 3: English
    if result.get("english"):
        english = result["english"]
    elif result.get("translation_error"):
        english = f"Translation not available:\n{result['translation_error']}"
    else:
        english = "Translation not performed."

    # Tab 4: Notes
    notes_parts = []
    if result.get("text_type"):
        notes_parts.append(f"**Text type:** {result['text_type'].title()}")
    if result.get("translation_confidence"):
        notes_parts.append(f"**Translation confidence:** {result['translation_confidence'].title()}")
    if result.get("notes"):
        notes_parts.append(f"\n**Scholarly Notes:**\n{result['notes']}")
    if result.get("iast"):
        notes_parts.append(f"\n**IAST Romanization:**\n`{result['iast']}`")
    notes_output = "\n\n".join(notes_parts) if notes_parts else "No notes available."

    # Tab 5: Flagged characters
    flagged = result.get("flagged_chars", [])
    if flagged:
        flagged_lines = ["**Low-confidence characters (OCR uncertainty):**\n"]
        for item in flagged:
            alts = ""
            if item.get("top5"):
                alt_list = [f"{c} ({p:.0%})" for c, p in item["top5"][:3]]
                alts = "  Alternatives: " + ", ".join(alt_list)
            flagged_lines.append(
                f"• `{item['file']}` → **{item['predicted']}** "
                f"({item['confidence']:.0%} confidence){alts}"
            )
        flagged_output = "\n".join(flagged_lines)
    else:
        flagged_output = "✅ All characters recognized with high confidence."

    # Stats card (HTML)
    stats_html = build_stats_html(result)

    progress(1.0, desc="Done!")

    return (
        nepal_bhasa,
        nepali,
        english,
        notes_output,
        flagged_output,
        stats_html,
    )


# ═══════════════════════════════════════════════════════════════════
# STATS HTML CARD
# ═══════════════════════════════════════════════════════════════════

def build_stats_html(result):
    """Build a small HTML stats summary card."""
    if result is None:
        return "<p style='color: gray; font-style: italic;'>No results yet. Upload an image and click Transliterate.</p>"

    conf_color = (
        "#22c55e" if result["avg_confidence"] > 0.85
        else "#f59e0b" if result["avg_confidence"] > 0.60
        else "#ef4444"
    )

    trans_status = "✅ Done" if result.get("nepali") else "⚠️ No API key"

    return f"""
<div style="
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    padding: 8px 0;
    font-family: sans-serif;
">
  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; text-align:center;">
    <div style="font-size:22px; font-weight:600; color:#1e293b;">{result["num_characters"]}</div>
    <div style="font-size:12px; color:#64748b; margin-top:2px;">Characters</div>
  </div>
  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; text-align:center;">
    <div style="font-size:22px; font-weight:600; color:#1e293b;">{result["num_lines"]}</div>
    <div style="font-size:12px; color:#64748b; margin-top:2px;">Lines</div>
  </div>
  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; text-align:center;">
    <div style="font-size:22px; font-weight:600; color:{conf_color};">{result["avg_confidence"]:.0%}</div>
    <div style="font-size:12px; color:#64748b; margin-top:2px;">Avg OCR Conf.</div>
  </div>
  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; text-align:center;">
    <div style="font-size:22px; font-weight:600; color:#1e293b;">{result["low_conf_count"]}</div>
    <div style="font-size:12px; color:#64748b; margin-top:2px;">Flagged</div>
  </div>
  <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:12px; text-align:center;">
    <div style="font-size:16px; font-weight:600; color:#1e293b;">{trans_status}</div>
    <div style="font-size:12px; color:#64748b; margin-top:2px;">Translation</div>
  </div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════
# GRADIO UI LAYOUT
# ═══════════════════════════════════════════════════════════════════

def build_ui():
    """Build and return the Gradio Blocks UI."""

    with gr.Blocks(
        title="Newa Script Transliteration",
        theme=gr.themes.Soft(
            primary_hue="slate",
            secondary_hue="blue",
        ),
        css="""
            .result-text { font-size: 18px; line-height: 1.8; }
            .devanagari  { font-size: 22px; line-height: 2.0; }
            footer { display: none !important; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────────
        gr.Markdown("""
# 𑑁 Newa Script Transliteration System
**Thesis Project** | Kathmandu Valley Manuscript OCR  
Upload a Newa manuscript image to get the text in **Nepal Bhasa (Devanagari)**, **Modern Nepali**, and **English**.
        """)

        # ── Main layout: Left controls | Right results ─────────────
        with gr.Row(equal_height=False):

            # ── LEFT: Controls ─────────────────────────────────────
            with gr.Column(scale=1, min_width=320):

                gr.Markdown("### 📷 Input")

                image_input = gr.Image(
                    label="Manuscript Image",
                    type="numpy",
                    sources=["upload", "clipboard"],
                    height=280,
                )

                gr.Markdown("### ⚙️ Settings")

                checkpoint_input = gr.Textbox(
                    label="Model Checkpoint Path",
                    value=DEFAULT_CHECKPOINT,
                    placeholder="checkpoints/best_model.pth",
                    info="Path to your trained best_model.pth file",
                )

                api_key_input = gr.Textbox(
                    label="Anthropic API Key (for translation)",
                    placeholder="sk-ant-api03-...",
                    type="password",
                    info="Required for Nepali + English translation. Get one at console.anthropic.com",
                )

                confidence_slider = gr.Slider(
                    label="OCR Confidence Threshold",
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    info="Characters below this confidence are flagged with ⟨?⟩",
                )

                with gr.Row():
                    keep_segments = gr.Checkbox(
                        label="Keep character crops",
                        value=False,
                        info="Save individual character images to output_segments/",
                    )
                    debug_mode = gr.Checkbox(
                        label="Debug mode",
                        value=False,
                        info="Save segmentation overlay image",
                    )

                run_button = gr.Button(
                    "🔍 Transliterate",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("""
---
**Pipeline steps:**
1. 🔲 Segment page → find characters
2. 🤖 OCR model → predict each character
3. 📝 Map to Devanagari letters
4. 🌐 Claude API → translate to Nepali + English
                """)

            # ── RIGHT: Results ─────────────────────────────────────
            with gr.Column(scale=2):

                gr.Markdown("### 📊 Recognition Statistics")
                stats_output = gr.HTML(
                    build_stats_html(None),
                    label="Stats",
                )

                gr.Markdown("### 📄 Results")

                with gr.Tabs():

                    # Tab 1: Nepal Bhasa (Devanagari)
                    with gr.Tab("𑑁 Nepal Bhasa (Devanagari)"):
                        gr.Markdown(
                            "_This is what the OCR model read from the manuscript — "
                            "the original Nepal Bhasa language written in Devanagari letters "
                            "instead of the original Newa script._"
                        )
                        nepal_bhasa_output = gr.Textbox(
                            label="Nepal Bhasa in Devanagari",
                            lines=10,
                            show_copy_button=True,
                            elem_classes=["devanagari"],
                            placeholder="OCR result will appear here...",
                        )

                    # Tab 2: Nepali
                    with gr.Tab("🇳🇵 Modern Nepali"):
                        gr.Markdown(
                            "_Modern Nepali translation of the classical Nepal Bhasa text. "
                            "Requires Anthropic API key._"
                        )
                        nepali_output = gr.Textbox(
                            label="Modern Nepali Translation",
                            lines=10,
                            show_copy_button=True,
                            elem_classes=["result-text"],
                            placeholder="Enter API key and click Transliterate...",
                        )

                    # Tab 3: English
                    with gr.Tab("🌏 English"):
                        gr.Markdown(
                            "_English translation of the manuscript text. "
                            "Requires Anthropic API key._"
                        )
                        english_output = gr.Textbox(
                            label="English Translation",
                            lines=10,
                            show_copy_button=True,
                            elem_classes=["result-text"],
                            placeholder="Enter API key and click Transliterate...",
                        )

                    # Tab 4: Notes
                    with gr.Tab("📚 Notes & IAST"):
                        gr.Markdown(
                            "_Scholarly notes about the text, historical context, "
                            "text type, and IAST romanization._"
                        )
                        notes_output = gr.Markdown(
                            value="Notes will appear here after transliteration.",
                        )

                    # Tab 5: Flagged characters
                    with gr.Tab("⚠️ Flagged"):
                        gr.Markdown(
                            "_Characters where the OCR model was uncertain. "
                            "Review these manually for accuracy._"
                        )
                        flagged_output = gr.Markdown(
                            value="Flagged characters will appear here.",
                        )

        # ── Wire up the button ─────────────────────────────────────
        run_button.click(
            fn=transliterate,
            inputs=[
                image_input,
                checkpoint_input,
                api_key_input,
                confidence_slider,
                keep_segments,
                debug_mode,
            ],
            outputs=[
                nepal_bhasa_output,
                nepali_output,
                english_output,
                notes_output,
                flagged_output,
                stats_output,
            ],
            show_progress="full",
        )

        # ── Footer info ────────────────────────────────────────────
        gr.Markdown("""
---
**About this system** | Newa Script OCR Thesis | Kathmandu Valley Manuscript Digitisation  
Model: CNN trained on 82 Newa character classes | Translation: Claude Sonnet API
        """)

    return demo


# ═══════════════════════════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Newa Transliteration Web UI")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                   help="Default checkpoint path shown in UI")
    p.add_argument("--port",   type=int, default=7860)
    p.add_argument("--share",  action="store_true",
                   help="Create a public shareable link (useful for thesis demos)")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't open browser automatically")
    args = p.parse_args()

    DEFAULT_CHECKPOINT = args.checkpoint

    print("\n" + "═" * 55)
    print("  NEWA MANUSCRIPT TRANSLITERATION UI")
    print("═" * 55)
    print(f"  Checkpoint: {DEFAULT_CHECKPOINT}")
    print(f"  Port:       {args.port}")
    if args.share:
        print("  Public URL: will be printed below")
    print("═" * 55 + "\n")

    demo = build_ui()
    demo.launch(
        server_port  = args.port,
        share        = args.share,
        inbrowser    = not args.no_browser,
        show_error   = True,
    )