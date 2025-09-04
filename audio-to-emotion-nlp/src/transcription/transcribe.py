"""Translate Polish sentences to English using a pretrained translation model."""

import time
from typing import Any

import pandas as pd
import requests

from classification.emotion import classify_emotions
from config.config import get_headers
from utils.helpers import format_time
from utils.logger import get_logger

logger = get_logger(__name__)


def request_transcription(upload_url: str) -> str:
    """Submit a transcription job to AssemblyAI and wait until it completes.

    This function polls the AssemblyAI API until the transcription status is
    'completed' or 'error', and returns the transcript ID on success.

    Args:
        upload_url: URL of the uploaded audio file to be transcribed.

    Returns:
        Transcript ID used to fetch the final transcript data.

    Raises:
        Exception: If transcription fails or a request error occurs.
    """
    endpoint = "https://api.assemblyai.com/v2/transcript"
    payload: dict[str, Any] = {
        "audio_url": upload_url,
        "language_code": "en",
        "punctuate": True,
        "format_text": True,
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=get_headers())
        resp.raise_for_status()
        transcript_id = resp.json()["id"]
        polling_url = f"{endpoint}/{transcript_id}"

        while True:
            poll = requests.get(polling_url, headers=get_headers()).json()
            status = poll.get("status")
            if status == "completed":
                logger.info("Transcription complete.")
                return transcript_id
            if status == "error":
                error_msg = poll.get("error", "unknown error")
                logger.error("Transcription failed: %s", error_msg)
                raise Exception(f"Transcription failed: {error_msg}")
            time.sleep(5)

    except requests.RequestException as exc:
        logger.error("Error during transcription request: %s", exc)
        raise Exception(f"Request failed: {exc}") from exc


def get_transcript_df(transcript_id: str) -> pd.DataFrame:
    """Retrieve sentence-level transcript and classify each sentence’s emotion.

    Args:
        transcript_id: ID of the completed transcription job.

    Returns:
        DataFrame with columns:
          - "Start Time": HH:MM:SS,mmm
          - "End Time": HH:MM:SS,mmm
          - "Sentence": Transcribed text
          - "Emotion": Predicted emotion label

    Raises:
        None—on error, returns an empty DataFrame.
    """
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences"

    try:
        resp = requests.get(endpoint, headers=get_headers())
        resp.raise_for_status()
        sentences = resp.json().get("sentences", [])

        if not sentences:
            logger.warning("No sentences found for transcript ID: %s", transcript_id)
            return pd.DataFrame()

        rows = []
        for s in sentences:
            rows.append(
                {
                    "Start Time": format_time(s["start"]),
                    "End Time": format_time(s["end"]),
                    "Sentence": s["text"],
                }
            )

        df = pd.DataFrame(rows)
        logger.debug("Built transcript DataFrame with %d rows", len(df))

        # Classify emotions
        df["Emotion"] = classify_emotions(df["Sentence"].tolist())
        logger.info(
            "Retrieved %d sentences for transcript ID: %s",
            len(df),
            transcript_id,
        )
        return df

    except requests.RequestException as exc:
        logger.error("Error retrieving transcript: %s", exc)
        return pd.DataFrame()
