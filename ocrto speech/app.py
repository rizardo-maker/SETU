import gradio as gr
from pathlib import Path
from setu_pipeline import SETUPipeline
from config import OUTPUT_AUDIO_DIR

# Global pipeline instance
pipeline = SETUPipeline(auto_download_models=True)

def process_file_ui(file_obj):
    if file_obj is None:
        return "Please upload a document or image file.", "", None, "No chunks generated."

    file_path = file_obj.name
    res = pipeline.process_file(file_path, synthesize_audio=True)

    status_str = f"**File:** {res['file_name']}  \n" \
                 f"**Type:** {res['file_type']} | **Scanned:** {res['is_scanned']}  \n" \
                 f"**OCR Engine:** {res['ocr_engine']}  \n" \
                 f"**TTS Engine:** {res['tts_engine']}  \n" \
                 f"**Processing Time:** OCR: {res['metrics']['ocr_time_seconds']}s | TTS: {res['metrics']['tts_time_seconds']}s | Total: {res['metrics']['total_time_seconds']}s"

    full_text = res['full_text']
    audio_files = res.get('audio_files', [])
    
    first_audio = audio_files[0] if audio_files else None

    chunks_summary = ""
    for idx, (chunk, audio_path) in enumerate(zip(res['chunks'], res['audio_files']), 1):
        chunks_summary += f"### Chunk {idx}\n**Text:** {chunk}\n**Audio File:** `{Path(audio_path).name}`\n\n---\n"

    return status_str, full_text, first_audio, chunks_summary

def voice_cmd_read_this():
    res = pipeline.read_first_chunk(play_speaker=False)
    if "error" in res:
        return res["error"], None
    status = f"🎙 Spoken Chunk 1 / {res['total_chunks']}"
    return f"**{status}**\n\nText: {res['text']}", res['audio_file']

def voice_cmd_continue():
    res = pipeline.read_next_chunk(play_speaker=False)
    if "error" in res:
        return res["error"], None
    status = f"🎙 Spoken Chunk {res['chunk_idx']} / {res['total_chunks']}"
    return f"**{status}**\n\nText: {res['text']}", res['audio_file']

with gr.Blocks(title="SETU: Offline OCR to Speech Pipeline") as demo:
    gr.Markdown("# 👁️ SETU: Offline Document OCR to English Speech Pipeline 🔊")
    gr.Markdown("100% Local Assistive Pipeline: PyMuPDF + RapidOCR / PaddleOCR + Piper Neural TTS (ONNX)")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Image (.png, .jpg) or PDF Document (.pdf)")
            process_btn = gr.Button("⚡ Process Document & Synthesize Speech", variant="primary")
            
            gr.Markdown("### 🎙 Simulated Voice Controls for Assistive Use")
            with gr.Row():
                read_btn = gr.Button("🎙 'Read This' (First Chunk)")
                continue_btn = gr.Button("⏭ 'Continue' (Next Chunk)")

        with gr.Column(scale=2):
            status_box = gr.Markdown("### Pipeline Status & Execution Metrics")
            text_output = gr.Textbox(label="Extracted Text", lines=8)
            audio_output = gr.Audio(label="Audio Speech Output (.wav)", type="filepath")

    with gr.Row():
        chunks_markdown = gr.Markdown(label="Chunked Speech Breakdown")

    # Event Handlers
    process_btn.click(
        fn=process_file_ui,
        inputs=[file_input],
        outputs=[status_box, text_output, audio_output, chunks_markdown]
    )

    read_btn.click(
        fn=voice_cmd_read_this,
        inputs=[],
        outputs=[status_box, audio_output]
    )

    continue_btn.click(
        fn=voice_cmd_continue,
        inputs=[],
        outputs=[status_box, audio_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
