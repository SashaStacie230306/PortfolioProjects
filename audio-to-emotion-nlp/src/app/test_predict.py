"""
test_predict.py
---------------
Standalone test script to run the audio-to-emotion pipeline using a YouTube link.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import requests
import yt_dlp
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

from app.predict import run_full_pipeline

# Download audio from YouTube
youtube_url = "https://www.youtube.com/watch?v=KAWvDsghyc8"
output_path = "temp_uploads/audio.mp3"

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([youtube_url])

# Upload audio to AssemblyAI
with open(output_path, "rb") as f:
    response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers={"authorization": os.getenv("ASSEMBLYAI_API_KEY")},
        files={"file": f},
    )

audio_url = response.json()["upload_url"]

# Run the full pipeline
df, paths = run_full_pipeline(audio_url=audio_url, language="auto")
print(df.head())
print("Saved CSV paths:", paths)
