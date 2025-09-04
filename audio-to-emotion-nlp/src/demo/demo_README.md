
# Demo — Gradio UI for SenseAI

This folder (`src/demo/`) contains a user-facing demo interface and a supporting API client for interacting with the **SenseAI** emotion classification system. It supports predictions from text, audio, video files, and online URLs (e.g., YouTube links).

---

## Contents

### `app_ui.py`
A modern, responsive **Gradio web app** that:
- Allows users to input text or upload audio/video files
- Accepts media URLs for transcription + emotion prediction
- Displays top emotion predictions in a table
- Enables CSV downloads of results
- Logs usage events (e.g., input submitted) to `frontend_usage.log`

To launch the UI locally:
```bash
python src/demo/app_ui.py
```

The app will open in your browser automatically.

---

### `api_client.py`
A backend-friendly Python client that:
- Sends requests to the FastAPI server
- Supports `text`, `audio`, `video`, and `URL` predictions
- Returns predictions as strings or `pandas.DataFrame`
- Logs all events (text submission, file upload, etc.)

All logs are saved to `frontend_usage.log` for later analysis.

---

## API Integration

The client reads the `API_URL` from your `.env` file or defaults to:

```env
API_URL=http://localhost:8000
```

Ensure your FastAPI backend is running at that URL before using the UI.

---

## Features

| Input Type    | Supported | Output Format        |
|---------------|-----------|----------------------|
| Text          | ✅         | Label + confidence   |
| Audio (.mp3)  | ✅         | Table + CSV          |
| Video (.mp4)  | ✅         | Table + CSV          |
| Online Media  | ✅         | Table + CSV          |

---

## Environment Setup

The Gradio demo requires the following dependencies:

- `gradio`
- `pandas`
- `requests`
- `python-dotenv`

Install them via your environment:
```bash
conda activate nlp7
```

Or directly via pip:
```bash
pip install gradio pandas requests python-dotenv
```

---

## Logging

All usage events are written to:
```
frontend_usage.log
```

This includes:
- Text input submissions
- File uploads (audio/video)
- Media URL predictions

---

## Summary

| File          | Purpose                                      |
|---------------|----------------------------------------------|
| `app_ui.py`   | Gradio-based frontend for emotion insights   |
| `api_client.py` | Client helper for calling FastAPI endpoints |
| `frontend_usage.log` | Logs user interactions and errors     |

This folder is your main gateway for demoing the full capabilities of the SenseAI system.

---
