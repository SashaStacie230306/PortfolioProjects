"""Gradio UI for SenseAI – a FastAPI-based multimodal emotion insight tool.

Supports text, audio, video file, and media URL (e.g., YouTube) inputs. Displays
prediction results and allows CSV download.
"""

import re
from pathlib import Path

import gradio as gr
import pandas as pd
from api_client import classify_audio, classify_text, classify_url, classify_video


def is_valid_url(url):
    """Check if the input string is a valid URL."""
    return re.match(r"^https?://", url) is not None


def main():
    """Main function to launch the Gradio UI."""
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f9fafb;
        padding: 20px;
    }
    h1 {
        font-size: 42px !important;
        text-align: center;
        color: #1f2937;
        margin-bottom: 10px;
    }
    h2 {
        text-align: center;
        font-size: 20px;
        color: #4b5563;
        margin-bottom: 30px;
    }
   .predict-button {
    background-color: #E26F66 !important;
    color: #FAFAFA !important;
    border-radius: 6px !important;
    padding: 10px 20px;
    font-weight: bold;
    border: none;
}
.predict-button:hover {
    background-color: #E26F66 !important;
    color: #FAFAFA !important;
}
    @media (max-width: 768px) {
        .gr-dataframe table {
            font-size: 12px !important;
        }
    }
    .gr-dataframe table {
      width: 100% !important;
        overflow-x: auto;
        white-space: nowrap;
        display: block;
    }
    """

    with gr.Blocks(css=custom_css, title="SenseAI") as demo:
        gr.Markdown("<h1> SenseAI</h1>")
        gr.Markdown(
            "<h2>AI-powered emotion insights across voice, text, and online content.</h2>"
        )

        with gr.Tabs():
            # === TEXT TAB ===
            with gr.Tab("Text Input"):
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Enter text",
                        lines=4,
                        placeholder="Type your sentence here...",
                    )
                    predict_btn = gr.Button(
                        "Predict Emotion", elem_classes="predict-button"
                    )
                    with gr.Row():
                        text_output_emotion = gr.Textbox(
                            label="Predicted Emotion", interactive=False
                        )
                        text_output_conf = gr.Textbox(
                            label="Confidence", interactive=False
                        )

                    error_text = gr.Markdown(visible=False)
                    loader = gr.Markdown("\u231b Predicting...", visible=False)
                    download_text_btn = gr.HTML(visible=False)

                def handle_text_input(text):
                    print(">>> handle_text_input CALLED <<<")
                    if not text.strip():
                        return (
                            "",
                            "",
                            gr.update(visible=True, value="Please enter some text."),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    result = classify_text(text)
                    print("Received result from backend:", result)

                    if not isinstance(result, str):
                        return (
                            "",
                            "",
                            gr.update(
                                visible=True,
                                value=f"Unexpected backend response type: {type(result)}",
                            ),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if result.startswith("Error"):
                        return (
                            "",
                            "",
                            gr.update(visible=True, value=result),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    # result = "Emotion: happiness | Confidence: 0.95"
                    if "|" not in result:
                        return (
                            "",
                            "",
                            gr.update(
                                visible=True,
                                value=f"Unexpected backend response: '{result}'",
                            ),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    try:
                        emotion, confidence = result.split("|")
                    except ValueError:
                        return (
                            "",
                            "",
                            gr.update(
                                visible=True,
                                value=f"Malformed response from backend: '{result}'",
                            ),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    download_html = "<a href='http://localhost:3126/download_text_csv' target='_blank'><button class='predict-button'>Download CSV</button></a>"

                    return (
                        emotion.strip(),
                        confidence.strip(),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(value=download_html, visible=True),
                    )

                predict_btn.click(
                    fn=handle_text_input,
                    inputs=text_input,
                    outputs=[
                        text_output_emotion,
                        text_output_conf,
                        error_text,
                        loader,
                        download_text_btn,
                    ],
                )

            # === AUDIO TAB ===
            with gr.Tab("Audio File"):
                audio_input = gr.Audio(
                    label="Upload audio file (.mp3, .wav)", type="filepath"
                )
                audio_btn = gr.Button(
                    "Predict from Audio", elem_classes="predict-button"
                )
                with gr.Row():
                    audio_output = gr.Dataframe(label="Top 10 Predictions")
                audio_error = gr.Markdown(visible=False)
                audio_loader = gr.Markdown("\u231b Processing audio...", visible=False)
                download_audio_btn = gr.HTML(visible=False)

                def handle_audio(file):
                    if not file:
                        return (
                            None,
                            None,
                            gr.update(
                                visible=True, value="Please upload a valid audio file."
                            ),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                    print("Audio file received:", file)
                    df, csv_path = classify_audio(file)
                    print("Backend classify_audio returned:", type(df), df)

                    if isinstance(df, str) and df.startswith("Error"):
                        return (
                            None,
                            None,
                            gr.update(visible=True, value=df),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if isinstance(df, pd.DataFrame) and df.columns.tolist() == [
                        "Error"
                    ]:
                        error_msg = (
                            df["Error"].iloc[0]
                            if not df.empty
                            else "Unknown error occurred."
                        )
                        return (
                            None,
                            None,
                            gr.update(visible=True, value=error_msg),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if isinstance(csv_path, str):
                        filename = Path(csv_path).name
                    elif isinstance(csv_path, list) and csv_path:
                        filename = Path(csv_path[0]).name
                    else:
                        filename = ""

                    print("Final filename used in download link:", filename)

                    download_html = f"<a href='http://localhost:3126/download_csv_file?filename={filename}' target='_blank'><button class='predict-button'>Download CSV</button></a>"

                    return (
                        df,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(value=download_html, visible=True),
                    )

                audio_btn.click(
                    lambda: gr.update(visible=True),
                    inputs=[],
                    outputs=[audio_loader],
                    queue=False,
                )
                audio_btn.click(
                    fn=handle_audio,
                    inputs=audio_input,
                    outputs=[
                        audio_output,
                        audio_error,
                        audio_loader,
                        download_audio_btn,
                    ],
                )

            # === VIDEO TAB ===
            with gr.Tab("Video File"):
                video_input = gr.File(
                    label="Upload video file (.mp4)", file_types=[".mp4"]
                )
                video_btn = gr.Button(
                    "Predict from Video", elem_classes="predict-button"
                )
                with gr.Row():
                    video_output = gr.Dataframe(label="Top 10 Predictions")
                video_error = gr.Markdown(visible=False)
                video_loader = gr.Markdown("\u231b Processing video...", visible=False)
                download_video_btn = gr.HTML(visible=False)

                def handle_video(file):
                    if not file:
                        return (
                            None,
                            None,
                            gr.update(
                                visible=True, value="Please upload a valid video file."
                            ),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    print("Video file received:", file)
                    df, csv_path = classify_video(file)
                    print("Backend classify_video returned:", type(df), df)

                    if isinstance(df, str) and df.startswith("Error"):
                        return (
                            None,
                            None,
                            gr.update(visible=True, value=df),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if isinstance(df, pd.DataFrame) and df.columns.tolist() == [
                        "Error"
                    ]:
                        error_msg = (
                            df["Error"].iloc[0]
                            if not df.empty
                            else "Unknown error occurred."
                        )
                        return (
                            None,
                            None,
                            gr.update(visible=True, value=error_msg),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if isinstance(csv_path, str):
                        filename = Path(csv_path).name
                    elif isinstance(csv_path, list) and csv_path:
                        filename = Path(csv_path[0]).name
                    else:
                        filename = ""

                    print("Final filename used in download link:", filename)

                    download_html = f"<a href='http://localhost:3126/download_csv_file?filename={filename}' target='_blank'><button class='predict-button'>Download CSV</button></a>"

                    return (
                        df,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(value=download_html, visible=True),
                    )

                video_btn.click(
                    lambda: gr.update(visible=True),
                    inputs=[],
                    outputs=[video_loader],
                    queue=False,
                )
                video_btn.click(
                    fn=handle_video,
                    inputs=video_input,
                    outputs=[
                        video_output,
                        video_error,
                        video_loader,
                        download_video_btn,
                    ],
                )

            # === URL TAB ===
            with gr.Tab("Media URL"):
                url_input = gr.Textbox(label="Paste media URL (YouTube, MP3, MP4)")
                lang_input = gr.Dropdown(
                    label="Transcription Language",
                    choices=["auto", "en", "pl"],
                    value="auto",
                    info="Choose the spoken language for media transcription (auto = detect automatically)",
                )
                url_btn = gr.Button("Predict from URL", elem_classes="predict-button")
                with gr.Row():
                    url_output = gr.Dataframe(label="Top 10 Predictions")
                url_plot = gr.Plot(visible=False)
                url_error = gr.Markdown(visible=False)
                download_url_btn = gr.HTML(visible=False)

                def handle_url_input(url, lang):
                    if not is_valid_url(url):
                        return (
                            None,
                            None,
                            gr.update(visible=True, value="Invalid URL format."),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    print(f"Received URL: {url}, language: {lang}")
                    df, csv_path = classify_url(url, lang)
                    print("Backend classify_url returned:", type(df), df)

                    if isinstance(df, str) and df.startswith("Error"):
                        return (
                            None,
                            None,
                            gr.update(visible=True, value=df),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if isinstance(df, pd.DataFrame) and df.columns.tolist() == [
                        "Error"
                    ]:
                        error_msg = (
                            df["Error"].iloc[0]
                            if not df.empty
                            else "Unknown error occurred."
                        )
                        return (
                            None,
                            None,
                            gr.update(visible=True, value=error_msg),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )

                    if isinstance(csv_path, str):
                        filename = Path(csv_path).name
                    elif isinstance(csv_path, list) and csv_path:
                        filename = Path(csv_path[0]).name
                    else:
                        filename = ""

                    print("Final filename used in download link:", filename)

                    download_html = f"<a href='http://localhost:3126/download_csv_file?filename={filename}' target='_blank'><button class='predict-button'>Download CSV</button></a>"

                    return (
                        df,
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(value=download_html, visible=True),
                    )

                url_btn.click(lambda: gr.update(visible=True), inputs=[], queue=False)
                url_btn.click(
                    fn=handle_url_input,
                    inputs=[url_input, lang_input],
                    outputs=[url_output, url_plot, url_error, download_url_btn],
                )

    #   demo.launch(share=True, inbrowser=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
