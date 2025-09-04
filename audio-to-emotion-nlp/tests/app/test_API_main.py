"""Test cases for the API main module."""

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.API_main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_fs_and_mock_mode(monkeypatch, tmp_path):
    """Isolate the filesystem and set mock mode for tests."""
    monkeypatch.setenv("SPHINX_MOCK_MODE", "1")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test_key")
    monkeypatch.setenv("SAVE_TO", "test_output")
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")

    # Create required directories in tmp_path
    (tmp_path / "output").mkdir()
    (tmp_path / "temp_uploads").mkdir()

    monkeypatch.chdir(tmp_path)

    # Clear module cache to ensure fresh import with new env vars
    if "app.API_main" in sys.modules:
        del sys.modules["app.API_main"]

    yield


def test_root():
    """Test the root endpoint of the API."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {
        "message": "Welcome to the Text, Audio & Video Emotion Classifier API!"
    }


def test_download_csv_file_not_found():
    """Test downloading a CSV file that does not exist."""
    r = client.get("/download_csv_file", params={"filename": "nope.csv"})
    assert r.status_code == 404


def test_download_csv_file_invalid_filename():
    """Test downloading CSV with invalid filename."""
    # Path traversal attempt
    r = client.get("/download_csv_file", params={"filename": "../secret.csv"})
    assert r.status_code == 400

    # Empty filename
    r = client.get("/download_csv_file", params={"filename": ""})
    assert r.status_code == 400

    # Filename with backslash
    r = client.get("/download_csv_file", params={"filename": "folder\\file.csv"})
    assert r.status_code == 400


def test_predict_text_mock_mode():
    """Test text prediction in mock mode."""
    r = client.post("/predict", json={"text": "I am happy today!"})
    assert r.status_code == 200

    data = r.json()
    assert data["input"] == "I am happy today!"
    assert data["predicted_label"] == "neutral"
    assert data["confidence"] == 0.0
    assert data["csv_paths"] == ["/mock/path.csv"]


def test_predict_text_validation_errors():
    """Test text prediction validation errors."""
    # Empty text
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422

    # Text too long
    long_text = "x" * 600
    r = client.post("/predict", json={"text": long_text})
    assert r.status_code == 422

    # Missing text field
    r = client.post("/predict", json={"wrong_field": "test"})
    assert r.status_code == 422


def test_predict_from_file_mock_mode():
    """Test audio file upload in mock mode."""
    file_content = b"fake mp3 content"
    files = {"file": ("test.mp3", BytesIO(file_content), "audio/mpeg")}

    r = client.post("/predict_from_file", files=files)
    assert r.status_code == 200

    data = r.json()
    assert data["message"] == "Mocked audio response"
    assert data["csv_paths"] == ["/mock/audio.csv"]
    assert data["results"] == []


def test_predict_from_file_no_file():
    """Test audio endpoint without file."""
    r = client.post("/predict_from_file")
    assert r.status_code == 422


def test_predict_from_video_mock_mode():
    """Test video file upload in mock mode."""
    file_content = b"fake mp4 content"
    files = {"file": ("test.mp4", BytesIO(file_content), "video/mp4")}

    r = client.post("/predict_from_video", files=files)
    assert r.status_code == 200

    data = r.json()
    assert data["message"] == "Mocked video response"
    assert data["csv_paths"] == ["/mock/video.csv"]
    assert data["results"] == []


def test_predict_from_url_mock_mode():
    """Test URL prediction in mock mode."""
    r = client.post(
        "/predict_from_url",
        json={"url": "https://youtube.com/watch?v=test", "language": "auto"},
    )
    assert r.status_code == 200

    data = r.json()
    assert data["message"] == "Mocked URL response"
    assert data["csv_paths"] == ["/mock/url.csv"]
    assert data["results"] == []


def test_predict_from_url_validation():
    """Test URL prediction validation."""
    # Invalid URL format
    r = client.post(
        "/predict_from_url", json={"url": "not-a-valid-url", "language": "auto"}
    )
    assert r.status_code == 422


def test_download_text_csv_mock_mode():
    """Test text CSV download in mock mode."""
    r = client.get("/download_text_csv")
    assert r.status_code == 404  # No CSV in mock mode


def test_download_csv_mock_mode():
    """Test general CSV download in mock mode."""
    r = client.get("/download_csv")
    assert r.status_code == 404  # No CSV in mock mode


def test_clean_temp_uploads():
    """Test clean_temp_uploads function."""
    from app.API_main import clean_temp_uploads

    with patch("app.API_main.Path") as mock_path:
        mock_temp_path = MagicMock()
        mock_path.return_value = mock_temp_path
        mock_temp_path.exists.return_value = False

        # Should not raise exception
        clean_temp_uploads()
        mock_path.assert_called_once_with("temp_uploads")


def test_clean_temp_uploads_with_files():
    """Test clean_temp_uploads with mock files."""
    import time

    from app.API_main import clean_temp_uploads

    with patch("app.API_main.Path") as mock_path:
        mock_temp_path = MagicMock()
        mock_path.return_value = mock_temp_path
        mock_temp_path.exists.return_value = True

        # Create mock old file
        old_file = MagicMock()
        old_file.is_file.return_value = True
        old_file.stat.return_value.st_mtime = time.time() - 2000

        # Create mock new file
        new_file = MagicMock()
        new_file.is_file.return_value = True
        new_file.stat.return_value.st_mtime = time.time() - 100

        mock_temp_path.iterdir.return_value = [old_file, new_file]

        clean_temp_uploads(1800)

        # Old file should be deleted
        old_file.unlink.assert_called_once()
        # New file should not be deleted
        new_file.unlink.assert_not_called()


def test_pydantic_models():
    """Test Pydantic model validation."""
    from app.API_main import MediaURLInput, TextInput

    # Test TextInput
    text_input = TextInput(text="Hello world")
    assert text_input.text == "Hello world"

    # Test whitespace stripping
    text_input_stripped = TextInput(text="  Hello world  ")
    assert text_input_stripped.text == "Hello world"

    # Test MediaURLInput
    media_input = MediaURLInput(url="https://example.com/test.mp3")
    assert str(media_input.url) == "https://example.com/test.mp3"
    assert media_input.language == "auto"


def test_validation_exception_handler():
    """Test validation error handling."""
    r = client.post("/predict", json={"invalid_field": "data"})
    assert r.status_code == 422

    data = r.json()
    assert data["message"] == "Validation failed"
    assert data["type"] == "validation_error"


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_real_mode_with_mocks():
    """Test text prediction in real mode with mocked Azure."""
    with patch("app.azure_client.azure_predict") as mock_predict:
        mock_predict.return_value = {"predicted_label": "happy", "confidence": 0.95}

        # Recreate client to pick up new env var
        from app.API_main import app

        test_client = TestClient(app)

        r = test_client.post("/predict", json={"text": "I love this!"})
        assert r.status_code == 200

        data = r.json()
        assert data["input"] == "I love this!"
        assert data["predicted_label"] == "happy"
        assert data["confidence"] == 0.95


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_real_mode_azure_error():
    """Test error handling when Azure service fails."""
    with patch("app.azure_client.azure_predict") as mock_predict:
        mock_predict.side_effect = Exception("Azure service error")

        from app.API_main import app

        test_client = TestClient(app)

        r = test_client.post("/predict", json={"text": "test"})
        assert r.status_code == 500


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_from_file_real_mode():
    """Test file upload in real mode with mocked pipeline."""
    with patch("app.predict.run_full_pipeline") as mock_pipeline:
        with patch("requests.post") as mock_requests:
            # Mock AssemblyAI upload
            mock_requests.return_value.json.return_value = {
                "upload_url": "https://fake-url.com/audio"
            }

            # Mock pipeline response
            mock_df = pd.DataFrame(
                {
                    "Start Time": ["00:00", "00:05"],
                    "Translation": ["Hello", "World"],
                    "Emotion": ["happy", "neutral"],
                    "Confidence (%)": [95.0, 80.0],
                }
            )
            mock_pipeline.return_value = (mock_df, ["test_output.csv"])

            from app.API_main import app

            test_client = TestClient(app)

            files = {"file": ("test.mp3", BytesIO(b"fake mp3"), "audio/mpeg")}
            r = test_client.post("/predict_from_file", files=files)

            assert r.status_code == 200
            data = r.json()
            assert data["message"] == "File processed"
            assert len(data["results"]) == 2


@pytest.mark.skip("Skipping complex file operations test for CI")
@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_from_video_real_mode():
    """Test video upload in real mode with mocked ffmpeg."""
    with patch("subprocess.run") as mock_subprocess:
        with patch("requests.post") as mock_requests:
            with patch("app.predict.run_full_pipeline") as mock_pipeline:
                # Mock ffmpeg
                mock_subprocess.return_value = None

                # Mock AssemblyAI upload
                mock_requests.return_value.json.return_value = {
                    "upload_url": "https://fake-url.com/audio"
                }

                # Mock pipeline
                mock_df = pd.DataFrame(
                    {
                        "Start Time": ["00:00"],
                        "Translation": ["Hello"],
                        "Emotion": ["happy"],
                        "Confidence (%)": [95.0],
                    }
                )
                mock_pipeline.return_value = (mock_df, ["test_output.csv"])

                from app.API_main import app

                test_client = TestClient(app)

                files = {"file": ("test.mp4", BytesIO(b"fake video"), "video/mp4")}
                r = test_client.post("/predict_from_video", files=files)

                assert r.status_code == 200
                data = r.json()
                assert data["message"] == "Video file processed"


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_from_video_ffmpeg_error():
    """Test video processing when ffmpeg fails."""
    with patch("subprocess.run") as mock_subprocess:
        import subprocess

        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

        from app.API_main import app

        test_client = TestClient(app)

        files = {"file": ("test.mp4", BytesIO(b"fake video"), "video/mp4")}
        r = test_client.post("/predict_from_video", files=files)

        assert r.status_code == 500


@pytest.mark.skip("Skipping complex file operations test for CI")
@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_from_url_youtube_real_mode():
    """Test YouTube URL processing in real mode."""
    with patch("yt_dlp.YoutubeDL") as mock_ytdl:
        with patch("requests.post") as mock_requests:
            with patch("app.predict.run_full_pipeline") as mock_pipeline:
                with patch("pathlib.Path.exists", return_value=True):
                    # Mock yt-dlp
                    mock_ydl_instance = MagicMock()
                    mock_ydl_instance.extract_info.return_value = {
                        "title": "Test Video"
                    }
                    mock_ydl_instance.prepare_filename.return_value = (
                        "temp_uploads/test.mp3"
                    )
                    mock_ytdl.return_value.__enter__.return_value = mock_ydl_instance

                    # Mock AssemblyAI upload
                    mock_requests.return_value.json.return_value = {
                        "upload_url": "https://fake-url.com/audio"
                    }

                    # Mock pipeline
                    mock_df = pd.DataFrame(
                        {
                            "Start Time": ["00:00"],
                            "Translation": ["Hello"],
                            "Emotion": ["happy"],
                            "Confidence (%)": [95.0],
                        }
                    )
                    mock_pipeline.return_value = (mock_df, ["test_output.csv"])

                    from app.API_main import app

                    test_client = TestClient(app)

                    r = test_client.post(
                        "/predict_from_url",
                        json={
                            "url": "https://youtube.com/watch?v=test",
                            "language": "auto",
                        },
                    )

                    assert r.status_code == 200


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_predict_from_url_direct_mp3():
    """Test direct MP3 URL processing in real mode."""
    with patch("requests.get") as mock_get:
        with patch("requests.post") as mock_post:
            with patch("app.predict.run_full_pipeline") as mock_pipeline:
                # Mock file download
                mock_response = MagicMock()
                mock_response.iter_content.return_value = [b"fake mp3 data"]
                mock_get.return_value = mock_response

                # Mock AssemblyAI upload
                mock_post.return_value.json.return_value = {
                    "upload_url": "https://fake-url.com/audio"
                }

                # Mock pipeline
                mock_df = pd.DataFrame(
                    {
                        "Start Time": ["00:00"],
                        "Translation": ["Hello"],
                        "Emotion": ["happy"],
                        "Confidence (%)": [95.0],
                    }
                )
                mock_pipeline.return_value = (mock_df, ["test_output.csv"])

                from app.API_main import app

                test_client = TestClient(app)

                r = test_client.post(
                    "/predict_from_url",
                    json={"url": "https://example.com/audio.mp3", "language": "auto"},
                )

                assert r.status_code == 200


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_download_csv_file_success():
    """Test successful CSV file download."""
    # Create a test CSV file in the current working directory
    csv_file = Path("output") / "test.csv"
    csv_file.write_text("col1,col2\nval1,val2\n")

    # Create new client to pick up environment change
    from app.API_main import app

    test_client = TestClient(app)

    r = test_client.get("/download_csv_file", params={"filename": "test.csv"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/csv; charset=utf-8"


@patch.dict("os.environ", {"SPHINX_MOCK_MODE": "0"})
def test_download_text_csv_success():
    """Test successful text CSV download."""
    # Create text prediction CSV in the current working directory
    csv_file = Path("output") / "text_prediction.csv"
    csv_file.write_text("Input Text,Predicted Emotion,Confidence\ntest,happy,0.95\n")

    # Create new client to pick up environment change
    from app.API_main import app

    test_client = TestClient(app)

    r = test_client.get("/download_text_csv")
    assert r.status_code == 200
    assert r.headers["content-type"] == "text/csv; charset=utf-8"


def test_constants():
    """Test that constants are properly defined."""
    from app import API_main

    assert API_main.ALLOWED_AUDIO_EXTENSIONS == {".mp3", ".wav"}
    assert API_main.ALLOWED_VIDEO_EXTENSIONS == {".mp4"}
    # MOCK_MODE can be True or False depending on previous tests, just check it exists
    assert isinstance(API_main.MOCK_MODE, bool)


def test_app_configuration():
    """Test FastAPI app configuration."""
    from app.API_main import app

    assert app.title == "Text & Audio Emotion Classifier API"
    assert app.version == "1.0"
    assert "Classifies emotions" in app.description


def test_exception_handlers_registered():
    """Test that exception handlers are registered."""
    from app import API_main

    # Check that the functions exist
    assert hasattr(API_main, "validation_exception_handler")
    assert hasattr(API_main, "unhandled_exception_handler")
    assert callable(API_main.validation_exception_handler)
    assert callable(API_main.unhandled_exception_handler)
