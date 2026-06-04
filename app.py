"""
app.py  —  Newa Manuscript Transliterator Web UI
Run:  python app.py
Open: http://localhost:7860
"""

import gradio as gr
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "transliteration"))
from translate import run_pipeline, save_report

CHECKPOINT = "checkpoints/best_model.pth"


def process_image(
    image_path,
    do_translate,
    seg_threshold_str,
    confidence,
    single_line_str,
):
    if image_path is None:
        return "Please upload a manuscript image.", "", "", "", "", None

    if not Path(CHECKPOINT).exists():
        return (f"Checkpoint not found: {CHECKPOINT}\n"
                "Make sure you are running from your project root directory.", "", "", "", "", None)

    # Parse threshold
    valley_threshold = None
    if seg_threshold_str and seg_threshold_str.strip().lower() not in ("", "auto"):
        try:
            valley_threshold = float(seg_threshold_str)
        except ValueError:
            return "Invalid threshold. Use a number like 0.15 or leave blank for Auto.", "", "", "", "", None

    # Parse line selection
    single_line = None
    if single_line_str and single_line_str.strip() not in ("", "0", "all"):
        try:
            single_line = int(single_line_str.strip()) - 1  # convert 1-based to 0-based
            if single_line < 0:
                return "Line number must be 1 or greater.", "", "", "", "", None
        except ValueError:
            return "Invalid line number. Enter a number like 3, or leave blank for all lines.", "", "", "", "", None

    result = run_pipeline(
        image_path           = image_path,
        checkpoint_path      = CHECKPOINT,
        translate            = do_translate,
        keep_segments        = True,
        debug                = True,
        confidence_threshold = float(confidence),
        valley_threshold     = valley_threshold,
        single_line          = single_line,
    )

    if not result["success"]:
        return f"ERROR: {result['error']}", "", "", "", "", None

    save_report(result, image_path)

    # Status
    mode_str = f"  |  Line {single_line + 1} only" if single_line is not None else ""
    status = (
        f"{result['num_characters']} characters  |  "
        f"{result['num_lines']} lines total{mode_str}  |  "
        f"Avg confidence: {result['avg_confidence']:.1%}  |  "
        f"Low-conf: {result['low_conf_count']}"
    )

    # Per-line text (not shown in single-line mode)
    lines_text = ""
    if result.get("lines_devanagari") and not result.get("single_line_mode"):
        lines_text = "\n".join(
            f"Line {i+1:02d}: {line}"
            for i, line in enumerate(result["lines_devanagari"])
        )

    devanagari = result.get("nepal_bhasa_devanagari") or ""
    iast       = result.get("iast") or ""

    translation = ""
    if result.get("nepali"):
        translation += f"Nepali:\n{result['nepali']}\n\n"
    if result.get("english"):
        translation += f"English:\n{result['english']}\n\n"
    if result.get("notes"):
        translation += f"Notes:\n{result['notes']}"
    if result.get("translation_error") and not translation:
        translation = result["translation_error"]

    # Debug image
    debug_img = None
    stem = Path(image_path).stem
    debug_path = Path("output_segments") / stem / "debug_segmentation.jpg"
    if debug_path.exists():
        debug_img = str(debug_path)

    return status, lines_text, devanagari, iast, translation, debug_img


def build_ui():
    with gr.Blocks(title="Newa Manuscript Transliterator") as demo:

        gr.Markdown("# Newa Manuscript Transliterator")
        gr.Markdown(
            "OCR pipeline for historical Nepal Bhasa manuscripts written in Prachalit/Newa script. "
            "Translation uses free Google Translate (no API key needed)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Manuscript Image",
                    type="filepath",
                    height=320,
                )

                single_line_input = gr.Textbox(
                    label="Single line mode (optional)",
                    placeholder="e.g. 3  (leave blank to read all lines)",
                    info="Enter a line number to read only that line. "
                         "Run once first to see how many lines were detected, then re-run with a line number.",
                )

                do_translate = gr.Checkbox(
                    label="Translate to English and Nepali (free Google Translate)",
                    value=True,
                )

                with gr.Accordion("Advanced settings", open=False):
                    seg_threshold = gr.Textbox(
                        label="Line detection threshold",
                        placeholder="Auto (recommended)",
                        info="Leave blank for auto. Try 0.15 if all lines merge into one. "
                             "Try 0.25 if too many false splits.",
                    )
                    conf_threshold = gr.Slider(
                        label="Min OCR confidence",
                        minimum=0.05, maximum=0.95, step=0.05, value=0.25,
                        info="Characters below this confidence are shown as unknown.",
                    )

                run_btn = gr.Button("Run", variant="primary", size="lg")

            with gr.Column(scale=2):
                status_out = gr.Textbox(label="Status", lines=2, interactive=False)

                with gr.Tabs():
                    with gr.TabItem("Per-Line Devanagari"):
                        lines_out = gr.Textbox(
                            label="All lines",
                            lines=14,
                            interactive=False,
                        )
                    with gr.TabItem("Devanagari (full text)"):
                        deva_out = gr.Textbox(
                            label="Nepal Bhasa in Devanagari script",
                            lines=10,
                            interactive=False,
                        )
                    with gr.TabItem("IAST Romanization"):
                        iast_out = gr.Textbox(
                            label="IAST phonetic romanization",
                            lines=10,
                            interactive=False,
                        )
                    with gr.TabItem("Translation"):
                        trans_out = gr.Textbox(
                            label="Google Translate output",
                            lines=12,
                            interactive=False,
                        )
                    with gr.TabItem("Segmentation Debug"):
                        debug_img_out = gr.Image(
                            label="Detected panels (orange), lines (orange), characters (green)",
                            height=480,
                        )

        gr.Markdown(
            "**Tips:** "
            "Check the Segmentation Debug tab first if results look wrong — it shows exactly what was detected. "
            "OCR accuracy is limited by training data. "
            "Line 00 having many characters usually means the border was not fully stripped — "
            "check the debug image."
        )

        run_btn.click(
            fn=process_image,
            inputs=[image_input, do_translate, seg_threshold, conf_threshold, single_line_input],
            outputs=[status_out, lines_out, deva_out, iast_out, trans_out, debug_img_out],
        )

    return demo


if __name__ == "__main__":
    print("Starting Newa Manuscript Transliterator...")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"  {'Found' if Path(CHECKPOINT).exists() else 'NOT FOUND -- run from project root'}")
    demo = build_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        inbrowser   = True,
    )