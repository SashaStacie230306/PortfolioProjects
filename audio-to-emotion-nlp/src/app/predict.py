"""
predict.py
----------
Handles audio transcription, emotion inference, and CSV generation.

This module connects to the AssemblyAI API to transcribe audio, translates
Polish sentences if necessary, performs emotion classification using a
preloaded model, and exports the results to CSV files.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from app.inference import classify_emotion, translate_if_needed

# MOCK_MODE disables model and API execution for documentation builds (e.g., Sphinx)
MOCK_MODE = os.getenv("SPHINX_MOCK_MODE") == "1"

if not MOCK_MODE:
    from app.model_loader import get_model_components

    model, tokenizer, label_encoder = get_model_components()

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# AssemblyAI setup
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
HEADERS = {"authorization": ASSEMBLYAI_API_KEY}


def get_output_paths(
    filename: str = "final_predictions.csv", save_to: str | None = None
) -> list[Path]:
    """Generate output file paths based on the specified save location(s).

    Args:
        filename (str): Name of the output CSV file.
        save_to (str, optional): Comma-separated list of target directories ('output', 'desktop', 'downloads').

    Returns:
        list[Path]: List of file paths where the CSV should be saved.
    """
    raw = (save_to or os.getenv("SAVE_TO", "output")).lower()
    targets = [p.strip() for p in raw.split(",")]

    paths = []
    for loc in targets:
        if loc == "desktop":
            paths.append(Path.home() / "Desktop" / filename)
        elif loc == "downloads":
            paths.append(Path.home() / "Downloads" / filename)
        elif loc == "output":
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            paths.append(output_dir / filename)

    return paths


def format_time(ms: int) -> str:
    """Convert time from milliseconds to HH:MM:SS,mmm format.

    Args:
        ms (int): Time in milliseconds.

    Returns:
        str: Formatted time string.
    """
    seconds = ms / 1000
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int((s % 1) * 1000):03}"


def upload_audio(audio_url: str, language: str = "auto") -> tuple[str, str]:
    """Uploads audio to AssemblyAI for transcription.

    Args:
        audio_url (str): The URL of the audio file to transcribe.
        language (str, optional): Language code ('pl', 'en', or 'auto').
        Defaults to "auto".

    Returns:
        tuple: (language_code, transcript_id)
    """
    if MOCK_MODE:
        return "en", "mock_id"

    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=HEADERS,
        json={
            "audio_url": audio_url,
            "language_code": language if language in ["pl", "en"] else None,
            "language_detection": language == "auto",
            "punctuate": True,
            "format_text": True,
        },
    )
    response.raise_for_status()
    transcript_id = response.json()["id"]

    poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        result = requests.get(poll_url, headers=HEADERS).json()
        if result["status"] == "completed":
            return result["language_code"], transcript_id
        if result["status"] == "error":
            raise RuntimeError(result["error"])
        time.sleep(5)


def get_sentences(transcript_id: str) -> list[dict]:
    """Retrieve sentence-level transcription results from AssemblyAI.

    Args:
        transcript_id (str): The transcript ID from AssemblyAI.

    Returns:
        list[dict]: List of sentence dictionaries.
    """
    if MOCK_MODE:
        return [{"start": 0, "end": 2000, "text": "This is a mock sentence."}]
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()["sentences"]


def process_audio_prediction(
    audio_url: str, save_to: str = None, language: str = "auto"
) -> dict:
    """Transcribes audio, classifies emotions, and saves results to CSV.

    Args:
        audio_url (str): Audio file URL.
        save_to (str, optional): CSV save location(s).
        language (str, optional): Language code.

    Returns:
        dict: Results and CSV paths.
    """
    if MOCK_MODE:
        mock_result = [
            {
                "Start Time": "00:00:00,000",
                "End Time": "00:00:02,000",
                "Sentence": "This is a mock sentence.",
                "Translation": "This is a mock sentence.",
                "Emotion": "neutral",
                "Confidence (%)": 0.0,
            }
        ]
        return {"results": mock_result, "csv_paths": ["/mock/path"]}

    lang_code, transcript_id = upload_audio(audio_url, language)
    sentences = get_sentences(transcript_id)

    results = []
    for s in sentences:
        original = s["text"]
        logging.info(f"Original sentence: {original}")

        translated = translate_if_needed(original, lang_code)
        if translated != original:
            logging.info(f"Translated to English: {translated}")

        emotion, confidence = classify_emotion(translated)
        logging.info(f"Predicted Emotion: {emotion} ({confidence}%)")

        results.append(
            {
                "Start Time": format_time(s["start"]),
                "End Time": format_time(s["end"]),
                "Sentence": original,
                "Translation": translated,
                "Emotion": emotion,
                "Confidence (%)": confidence,
            }
        )

    df = pd.DataFrame(results)
    csv_path = get_output_paths("final_predictions.csv", save_to=save_to)

    for path in csv_path:
        df.to_csv(path, index=False)
        logging.info(f"CSV saved to: {path}")

    return {"results": results, "csv_paths": [str(p) for p in csv_path]}


def predict_label(text: str) -> dict:
    """Predict the emotion label and confidence for a given text.

    Args:
        text (str): The input text to classify.

    Returns:
        dict: Dictionary containing the predicted label and confidence.
        Or an error message if text is empty.
    """
    if MOCK_MODE:
        return {"predicted_label": "neutral", "confidence": 0.0}

    if not text.strip():
        return {"error": "Text is empty."}
    label, confidence = classify_emotion(text)
    return {"predicted_label": label, "confidence": confidence}


def run_full_pipeline(audio_url: str, save_to: str = None, language: str = "auto"):
    """Runs the pipeline and saves results to CSV.

    Args:
        audio_url (str): URL of the audio file to process.
        save_to (str, optional): Location(s) to save the CSV file.
        language (str, optional): Language code for transcription.

    Returns:
        tuple: DataFrame of results and list of CSV file paths.
    """
    result = process_audio_prediction(audio_url, save_to=save_to, language=language)
    return pd.DataFrame(result["results"]), result["csv_paths"]
