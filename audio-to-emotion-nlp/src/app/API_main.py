"""API_main.py.

FastAPI server exposing API endpoints for text, audio, and video.
"""

# Standard library
import os
import time
import urllib.parse
from os.path import basename
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Third-party packages
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl, constr
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from src.utils.logger import get_logger, setup_logging

# Load environment variables
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
MOCK_MODE = os.getenv("SPHINX_MOCK_MODE") == "1"

# Configure logging to file and console
setup_logging(log_filename="api.log")
logger = get_logger(__name__)


# Conditional imports
if not MOCK_MODE:
    import subprocess
    import uuid

    import requests
    import yt_dlp

    from app.azure_client import azure_predict
    from app.predict import get_output_paths, run_full_pipeline

    ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
    SAVE_TO = os.getenv("SAVE_TO", "output").lower()
    Path(SAVE_TO).mkdir(parents=True, exist_ok=True)
else:
    ASSEMBLYAI_API_KEY = ""
    SAVE_TO = "output"

# Constants
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}

# FastAPI app setup
app = FastAPI(
    title="Text & Audio Emotion Classifier API",
    description="Classifies emotions with timestamps, predictions, confidence scores.",
    version="1.0",
)
logger.debug("FastAPI app initialized")

print("FastAPI app initialized")

# Constrained string for text input
TextStr = constr(strip_whitespace=True, min_length=1, max_length=512)


# Response models
class TextInput(BaseModel):
    """Model for text input to the emotion classification API."""

    text: constr(strip_whitespace=True, min_length=1, max_length=512)


class MediaURLInput(BaseModel):
    """Model for media URL input to the emotion classification API."""

    url: HttpUrl
    language: str = "auto"


class TextPredictionResponse(BaseModel):
    """Response model for text emotion prediction."""

    input: str
    predicted_label: str
    confidence: float
    csv_paths: List[str]


class ErrorResponse(BaseModel):
    """Model for error responses from the API."""

    message: str
    type: str
    detail: Optional[Dict] = None


class MediaPredictionResponse(BaseModel):
    """Response model for media (audio/video) emotion prediction."""

    message: str
    csv_paths: List[str]
    results: List[Dict]


# Error handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle FastAPI validation errors.

    Returns:
        JSONResponse: Details of the validation error.
    """
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "message": "Validation failed",
            "type": "validation_error",
            "detail": exc.errors(),
        },
    )


# Error handler for unexpected errors
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors in the API."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "Internal server error",
            "type": "server_error",
            "detail": str(exc),
        },
    )


# Utility: Cleanup old temp files
def clean_temp_uploads(older_than_seconds=1800):
    """Delete temporary files in the 'temp_uploads' directory.

    Args:
        older_than_seconds (int): Older files will be deleted.
            Defaults to 1800 seconds (30 minutes).

    Returns:
        None
    """
    logger.debug("Starting cleanup of temp_uploads directory")
    temp_path = Path("temp_uploads")
    if not temp_path.exists():
        logger.debug("No temp_uploads directory to clean")
        return
    for file in temp_path.iterdir():
        if file.is_file() and time.time() - file.stat().st_mtime > older_than_seconds:
            try:
                file.unlink()
                logger.info(f"Deleted old file: {file}")
            except Exception as e:
                logger.warning(f"Could not delete {file}: {e}")


# Root endpoint
@app.get("/")
def root() -> dict:
    """Root endpoint for checking API health status.

    Returns:
        dict: A simple welcome message indicating the API is online.
    """
    logger.info("Root endpoint called")
    return {"message": "Welcome to the Text, Audio & Video Emotion Classifier API!"}


# Text prediction endpoint
@app.post(
    "/predict",
    summary="Classify emotion from text",
    response_model=TextPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def predict(input: TextInput) -> TextPredictionResponse:
    """Predict the emotion of a given text string using a transformer model.

    Args:
        input (TextInput): A JSON object containing the input text.

    Returns:
        TextPredictionResponse: Predicted emotion label, confidence, and path output.
    """
    logger.info(f"Received /predict request with text: {input.text}")
    if MOCK_MODE:
        logger.debug("MOCK_MODE active - returning dummy prediction")
        return {
            "input": input.text,
            "predicted_label": "neutral",
            "confidence": 0.0,
            "csv_paths": ["/mock/path.csv"],
        }

    try:
        logger.info("Calling azure_predict for text classification")
        result = azure_predict(input.text)
        # If azure_predict somehow still returns an error dict:
        if isinstance(result, dict) and "error" in result:
            raise RuntimeError(result["error"])

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        df = pd.DataFrame(
            [
                {
                    "Input Text": input.text,
                    "Predicted Emotion": result["predicted_label"],
                    "Confidence": result["confidence"],
                }
            ]
        )
        df.to_csv(output_dir / "text_prediction.csv", index=False)
        logger.info(f"[DEBUG] Text received by API: '{input.text}'")

        return {
            "input": input.text,
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "csv_paths": ["/download_text_csv"],
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


# Audio file upload endpoint
@app.post(
    "/predict_from_file",
    summary="Upload audio file for emotion prediction",
    response_model=MediaPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_from_file(file: UploadFile = File(...)) -> MediaPredictionResponse:
    """Upload an audio file for transcription and emotion classification.

    Args:
        file (UploadFile): The uploaded .mp3 or .wav file.

    Returns:
        MediaPredictionResponse:Contains predicted emotions, timestamps,saved CSV paths.
    """
    logger.debug(f"/predict_from_file: Received file '{file.filename}'")
    if MOCK_MODE:
        logger.debug("MOCK_MODE active. Returning mock audio response.")
        return {
            "message": "Mocked audio response",
            "csv_paths": ["/mock/audio.csv"],
            "results": [],
        }

    try:
        contents = await file.read()
        file_path = Path("temp_uploads") / basename(file.filename)
        os.makedirs("temp_uploads", exist_ok=True)
        logger.debug(f"Saving uploaded file to: {file_path}")
        with open(file_path, "wb") as f:
            f.write(contents)

        logger.debug("Uploading file to AssemblyAI...")
        with open(file_path, "rb") as audio_file:
            upload_response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers={"authorization": ASSEMBLYAI_API_KEY},
                files={"file": audio_file},
            )
        os.remove(file_path)
        audio_url = upload_response.json()["upload_url"]
        logger.info("File uploaded to AssemblyAI successfully")

        final_df, csv_paths = run_full_pipeline(audio_url, save_to=SAVE_TO)
        logger.info("Pipeline completed for audio file")
        results = final_df[
            ["Start Time", "Translation", "Emotion", "Confidence (%)"]
        ].to_dict(orient="records")

        return {
            "message": "File processed",
            "csv_paths": [Path(p).name for p in csv_paths],
            "results": results,
        }
    except Exception as e:
        logger.error(f"Audio file processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Video file upload endpoint
@app.post(
    "/predict_from_video",
    summary="Upload video file for emotion prediction",
    response_model=MediaPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_from_video(file: UploadFile = File(...)) -> MediaPredictionResponse:
    """Upload a video file to extract audio, transcribe, and classify emotions.

    Args:
        file (UploadFile): The uploaded .mp4 video file.

    Returns:
        MediaPredictionResponse.
    """
    logger.debug(f"/predict_from_video called with file: {file.filename}")
    if MOCK_MODE:
        logger.debug("MOCK_MODE active - returning dummy video prediction")
        return {
            "message": "Mocked video response",
            "csv_paths": ["/mock/video.csv"],
            "results": [],
        }

    try:
        video_id = str(uuid.uuid4())
        input_path = f"temp_uploads/{video_id}.mp4"
        audio_path = f"temp_uploads/{video_id}.mp3"

        os.makedirs("temp_uploads", exist_ok=True)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        logger.debug(f"Saved uploaded video to {input_path}, extracting audio...")
        subprocess.run(
            ["ffmpeg", "-i", input_path, "-vn", "-acodec", "libmp3lame", audio_path],
            check=True,
        )
        logger.debug(f"Audio extracted to {audio_path}, uploading to AssemblyAI")

        with open(audio_path, "rb") as f:
            response = requests.post(
                "https://api.assemblyai.com/v2/upload",
                headers={"authorization": ASSEMBLYAI_API_KEY},
                files={"file": f},
            )
        audio_url = response.json()["upload_url"]

        os.remove(input_path)
        os.remove(audio_path)
        logger.info("Video file converted and uploaded")

        final_df, csv_paths = run_full_pipeline(audio_url, save_to=SAVE_TO)
        results = final_df[
            ["Start Time", "Translation", "Emotion", "Confidence (%)"]
        ].to_dict(orient="records")

        return {
            "message": "Video file processed",
            "csv_paths": [Path(p).name for p in csv_paths],
            "results": results,
        }

    except subprocess.CalledProcessError:
        logger.error("ffmpeg failed to convert video to audio")
        raise HTTPException(
            status_code=500, detail="ffmpeg failed to convert the video to audio."
        )
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Media URL prediction endpoint
@app.post(
    "/predict_from_url",
    summary="Classify emotion from a media URL (MP3/MP4/YouTube)",
    response_model=MediaPredictionResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def predict_from_url(input: MediaURLInput) -> MediaPredictionResponse:
    """Predict emotion from a media URL (YouTube, MP3, MP4)."""

    logger.debug(
        f"/predict_from_url called with URL: {input.url}, language: {input.language}"
    )
    if MOCK_MODE:
        logger.debug("MOCK_MODE active - returning dummy URL prediction")
        return {
            "message": "Mocked URL response",
            "csv_paths": ["/mock/url.csv"],
            "results": [],
        }

    try:
        clean_temp_uploads()
        media_url = str(input.url)

        # Strip query strings like &t=778s for YouTube links
        parsed = urllib.parse.urlparse(media_url)
        media_url = urllib.parse.urlunparse(parsed)

        logger.info(f"Downloading and processing media from URL: {media_url}")

        if "youtube.com" in media_url or "youtu.be" in media_url:
            output_dir = "temp_uploads"
            os.makedirs(output_dir, exist_ok=True)
            yt_id = str(uuid.uuid4())  # Unique filename prefix
            outtmpl = f"{output_dir}/{yt_id}.%(ext)s"

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": outtmpl,
                "postprocessors": [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "mp3",
                        "preferredquality": "192",
                    }
                ],
                "quiet": False,
                "noplaylist": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.debug("Starting YouTube download via yt_dlp")
                info = ydl.extract_info(media_url, download=True)
                logger.info(f"Downloaded title: {info.get('title')}")
                downloaded_filename = Path(ydl.prepare_filename(info)).with_suffix(
                    ".mp3"
                )

            if not downloaded_filename.exists():
                logger.error(f"Expected MP3 not found: {downloaded_filename}")
                raise HTTPException(
                    status_code=500, detail="Audio file was not extracted."
                )

            with open(downloaded_filename, "rb") as f:
                logger.debug(
                    f"Uploading extracted MP3 to AssemblyAI: {downloaded_filename}"
                )
                upload_response = requests.post(
                    "https://api.assemblyai.com/v2/upload",
                    headers={"authorization": ASSEMBLYAI_API_KEY},
                    files={"file": f},
                )
            media_url = upload_response.json()["upload_url"]
            os.remove(downloaded_filename)

        elif media_url.endswith(".mp3") or media_url.endswith(".mp4"):
            try:
                logger.info(f"Handling direct media file: {media_url}")
                response = requests.get(media_url, stream=True, timeout=10)
                response.raise_for_status()
                ext = media_url.split(".")[-1].split("?")[0]
                local_file = Path("temp_uploads") / f"temp_{uuid.uuid4()}.{ext}"
                with open(local_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.debug(f"Saved direct media file locally: {local_file}")

                with open(local_file, "rb") as f:
                    upload_response = requests.post(
                        "https://api.assemblyai.com/v2/upload",
                        headers={"authorization": ASSEMBLYAI_API_KEY},
                        files={"file": f},
                    )
                media_url = upload_response.json()["upload_url"]
                os.remove(local_file)
            except Exception as e:
                logger.error(f"Direct media download failed: {e}")
                raise HTTPException(
                    status_code=400, detail="Failed to handle direct media URL."
                )

        logger.debug("Running full prediction pipeline...")
        final_df, csv_paths = run_full_pipeline(
            media_url, save_to=SAVE_TO, language=input.language
        )

        required_columns = {"Start Time", "Translation", "Emotion", "Confidence (%)"}
        if not required_columns.issubset(set(final_df.columns)):
            logger.error(
                f"Missing expected columns in result dataframe: {final_df.columns}"
            )
            raise HTTPException(
                status_code=500, detail="Result data is incomplete or corrupted."
            )

        results = final_df[
            ["Start Time", "Translation", "Emotion", "Confidence (%)"]
        ].to_dict(orient="records")

        return {
            "message": "Media processed successfully",
            "csv_paths": [Path(p).name for p in csv_paths],
            "results": results,
        }

    except Exception as e:
        logger.exception("Media URL processing failed.")
        raise HTTPException(
            status_code=500, detail=f"Media URL processing failed: {str(e)}"
        )


logger.debug("Mounting '/output' static file directory for CSV access")
# Mount the output directory to serve CSVs directly
app.mount("/output", StaticFiles(directory="output"), name="output")


# CSV download endpoints
@app.get(
    "/download_csv",
    summary="Download CSV of last audio/video prediction (output/ only)",
)
def download_csv():
    """Download the most recent CSV file with audio or video."""
    logger.debug("/download_csv endpoint called")

    if MOCK_MODE:
        logger.warning("MOCK_MODE active - no CSV to download")
        raise HTTPException(status_code=404, detail="MOCK_MODE: CSV not available.")

    path = get_output_paths("final_predictions.csv", "output")[0]
    if path.exists():
        return FileResponse(
            str(path), media_type="text/csv", filename="final_predictions.csv"
        )
    raise HTTPException(status_code=404, detail="CSV not found.")


@app.get("/download_text_csv", summary="Download CSV of last text prediction")
def download_text_csv():
    """Download the most recent CSV file with text emotion prediction results."""
    logger.debug("/download_text_csv endpoint called")
    """Download the most recent CSV file with text-based emotion prediction results.

    Returns:
        FileResponse: CSV file containing classification output.
    """
    path = Path("output/text_prediction.csv")
    if MOCK_MODE:
        logger.warning("MOCK_MODE active - no text CSV to download")
        raise HTTPException(status_code=404, detail="MOCK_MODE: CSV not available.")

    if path.exists():
        logger.info(f"Serving audio/video prediction CSV: {path}")
        return FileResponse(
            str(path), media_type="text/csv", filename="text_prediction.csv"
        )
    logger.warning("CSV 'final_predictions.csv' not found in output directory")
    raise HTTPException(status_code=404, detail="CSV not found.")


@app.get("/download_csv_file")
def download_csv_file(
    filename: str = Query(..., description="CSV filename to download")
):
    """Download a specific CSV file from the output directory."""
    logger.debug(f"/download_csv_file endpoint called with filename: {filename}")

    if not filename or "/" in filename or "\\" in filename:
        logger.warning(f"Invalid filename requested: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = Path("output") / filename

    if file_path.exists() and file_path.is_file():
        logger.info(f"Serving requested CSV file: {file_path}")
        return FileResponse(
            file_path,
            media_type="text/csv",
            filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    logger.warning(f"Requested CSV file not found: {filename}")
    raise HTTPException(status_code=404, detail=f"CSV file '{filename}' not found.")


logger.info("API server initialized and ready to receive requests.")
# Run the app with: uvicorn API_main:app --reload
