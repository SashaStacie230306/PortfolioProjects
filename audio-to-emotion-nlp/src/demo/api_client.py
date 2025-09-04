import logging
import os
from logging.handlers import RotatingFileHandler

import pandas as pd
import requests

# Setup logging
log_handler = RotatingFileHandler(
    "frontend_usage.log", maxBytes=5_000_000, backupCount=3
)
formatter = logging.Formatter("%(asctime)s - %(message)s")
log_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

API_URL = os.getenv("API_URL", "http://backend:8000")


def log_event(event: str):
    """Log frontend usage events.

    Args:
        event (str): Description of the event.
    """
    logger.info(event)


def classify_text(text: str) -> str:
    """Classify emotion from text using FastAPI.

    Args:
        text (str): Input sentence.

    Returns:
        str: Predicted emotion and confidence.
    """
    log_event("Text input submitted")
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": text})
        response.raise_for_status()
        data = response.json()
        return f"{data['predicted_label']} | {data['confidence']:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"


def classify_audio(file_path: str):
    """Classify emotion from audio file.

    Args:
        file_path (str): Path to audio file.

    Returns:
        tuple: Top 10 predictions and path to full CSV file.
    """
    log_event("Audio file uploaded")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{API_URL}/predict_from_file", files={"file": f})
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["results"])
        filename = data.get("csv_paths", [])
        filename = filename[0] if filename else None
        return df.head(10), filename
    except Exception as e:
        error_df = pd.DataFrame(columns=["Error"])
        error_df.loc[0] = str(e)
        return error_df, None


def classify_video(file_path: str):
    """Classify emotion from video file.

    Args:
        file_path (str): Path to video file.

    Returns:
        tuple: Top 10 predictions and path to full CSV file.
    """
    log_event("Video file uploaded")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(f"{API_URL}/predict_from_video", files={"file": f})
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["results"])
        filename = data.get("csv_paths", [])
        filename = filename[0] if filename else None
        return df.head(10), filename
    except Exception as e:
        error_df = pd.DataFrame(columns=["Error"])
        error_df.loc[0] = str(e)
        return error_df, None


def classify_url(media_url: str, language: str = "auto"):
    """Classify emotion from media URL.

    Args:
        media_url (str): Media link (YouTube, MP3, MP4).
        language (str): Optional transcription language.

    Returns:
        tuple: Top 10 predictions and path to full CSV file.
    """
    log_event(f"URL submitted: {media_url}")
    try:
        payload = {"url": media_url, "language": language}
        response = requests.post(f"{API_URL}/predict_from_url", json=payload)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data["results"])
        filename = data.get("csv_paths", [])
        filename = filename[0] if filename else None
        return df.head(10), filename
    except Exception as e:
        error_df = pd.DataFrame(columns=["Error"])
        error_df.loc[0] = str(e)
        return error_df, None
