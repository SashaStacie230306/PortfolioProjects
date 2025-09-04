# Emotion Classifier API Test Suite

This directory contains the test suite for validating the functionality of the Emotion Classification API. It includes unit tests, integration tests, and validation logic for various components such as the FastAPI app, transcription services, classification models, and utility functions.

## Structure

* `tests/classification/`: Tests for the text emotion classification logic.
* `tests/app/`: Covers API endpoints and application setup.
* `tests/transcription/`: Verifies transcription workflows and integration.
* `tests/utils/`: Tests for helper utilities like logging and CSV handling.
* `tests/models/`: Covers CLI-based model loading and evaluation.

## Running Tests

To run all tests and measure code coverage, use:

```bash
PYTHONPATH=src poetry run pytest --cov=src --cov=app --cov-report=term-missing
```

To suppress warning messages during the test run:

```bash
PYTHONPATH=src poetry run pytest --disable-warnings
```

## Test Modes

The test suite supports both **mock mode** and **real mode**:

* **Mock Mode**: Enabled by setting `SPHINX_MOCK_MODE=1`. This bypasses external dependencies such as cloud services, useful for fast and isolated testing.
* **Real Mode**: Set `SPHINX_MOCK_MODE=0` to use real services (e.g., Azure, AssemblyAI) with mocked responses or test integrations.

## Test Categories

* **Unit Tests**: Focus on individual functions with mocked dependencies.
* **Integration Tests**: Validate the full pipeline, such as uploading files and processing audio/video/text through endpoints.
* **Validation Tests**: Ensure input/output schema, error handling, and environment variable logic work correctly.

## Notes

* Integration tests involving file uploads (e.g., `.mp3` and `.mp4`) and external APIs are partially skipped or mocked to maintain test speed and reliability.
* CSV handling and cleanup logic is tested to ensure correct file generation and isolation between test runs.
