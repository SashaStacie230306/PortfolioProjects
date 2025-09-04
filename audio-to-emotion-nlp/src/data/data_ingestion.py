"""Translate Polish sentences to English using a pretrained translation model."""

import requests

from src.config.config import get_headers
from utils.logger import get_logger

# module-scoped logger
logger = get_logger(__name__)


def upload_file(filepath: str) -> str:
    """Upload an audio file to the AssemblyAI transcription API.

    Args:
        filepath (str): Full path to the local audio file (e.g. .mp3 or .wav).

    Returns:
        str: Publicly accessible upload URL returned by the API.

    Raises:
        Exception: If the upload fails due to a bad response or other issue.
    """
    try:
        with open(filepath, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers=get_headers(),
                files={"file": f},
            )
            if response.status_code == 200:
                logger.info("Uploaded: %s", filepath)
                return response.json()["upload_url"]
            else:
                logger.error("Upload failed: %s", response.text)
                raise Exception(f"Upload failed: {response.text}")
    except Exception as e:
        logger.error("Failed to upload file %s: %s", filepath, e)
        raise e
