"""Unit tests for src/transcription/transcribe.py."""

from types import SimpleNamespace

import pandas as pd
import pytest
import requests

# Import the real module under test
import transcription.transcribe as transcribe


@pytest.fixture(autouse=True)
def disable_sleep(monkeypatch):
    """Prevent actual sleeping in polling loops."""
    monkeypatch.setattr(transcribe, "time", SimpleNamespace(sleep=lambda *_: None))


@pytest.fixture(autouse=True)
def stub_get_headers(monkeypatch):
    """Stub get_headers to return a dummy header dict."""
    monkeypatch.setattr(
        "transcription.transcribe.get_headers",
        lambda: {"authorization": "test"},
    )


def test_request_transcription_success(monkeypatch):
    """Should poll until completed and return transcript ID."""
    fake_id = "abc123"

    def fake_post(url, json, headers):
        assert url.endswith("/v2/transcript")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": fake_id},
        )

    calls = {"count": 0}

    def fake_get(url, headers):
        calls["count"] += 1
        if calls["count"] == 1:
            return SimpleNamespace(json=lambda: {"status": "processing"})
        return SimpleNamespace(json=lambda: {"status": "completed"})

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    tid = transcribe.request_transcription("http://audio.mp3")
    assert tid == fake_id


def test_request_transcription_error_status(monkeypatch):
    """If polling returns error status, should raise Exception."""
    fake_id = "err123"
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, json, headers: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": fake_id},
        ),
    )
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers: SimpleNamespace(
            json=lambda: {"status": "error", "error": "bad"}
        ),
    )

    with pytest.raises(Exception) as exc:
        transcribe.request_transcription("u")
    assert "Transcription failed: bad" in str(exc.value)


def test_request_transcription_request_exception(monkeypatch):
    """If requests.post raises RequestException, should wrap and raise."""
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.RequestException("fail")
        ),
    )
    with pytest.raises(Exception) as exc:
        transcribe.request_transcription("u")
    assert "Request failed: fail" in str(exc.value)


def test_get_transcript_df_empty(monkeypatch):
    """Empty 'sentences' yields empty DataFrame."""
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"sentences": []},
        ),
    )
    # Ensure classify_emotions is not called
    monkeypatch.setattr(
        transcribe,
        "classify_emotions",
        lambda *_: (_ for _ in ()).throw(AssertionError),
    )
    df = transcribe.get_transcript_df("tid")
    assert isinstance(df, pd.DataFrame) and df.empty


def test_get_transcript_df_success(monkeypatch):
    """Returns DataFrame with correct columns and calls classify_emotions."""
    sentences = [
        {"start": 0, "end": 1000, "text": "hi"},
        {"start": 1000, "end": 2000, "text": "bye"},
    ]
    monkeypatch.setattr(
        requests,
        "get",
        lambda url, headers: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"sentences": sentences},
        ),
    )
    monkeypatch.setattr(transcribe, "classify_emotions", lambda texts: ["happy", "sad"])
    monkeypatch.setattr(transcribe, "format_time", lambda ms: f"t{ms}")

    df = transcribe.get_transcript_df("tid")
    assert list(df.columns) == ["Start Time", "End Time", "Sentence", "Emotion"]
    assert df.loc[0, "Emotion"] == "happy"
    assert df.loc[1, "Emotion"] == "sad"


def test_get_transcript_df_request_exception(monkeypatch):
    """If GET raises RequestException, return empty DataFrame."""
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            requests.RequestException("oops")
        ),
    )
    df = transcribe.get_transcript_df("tid")
    assert isinstance(df, pd.DataFrame) and df.empty
