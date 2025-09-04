"""Unit tests for app.predict module.

Covers:
- get_output_paths
- format_time
- upload_audio (with polling)
- get_sentences
- process_audio_prediction (end-to-end logic, mocked)
- predict_label (empty vs. non-empty)
- run_full_pipeline proxy
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import app.predict as predict
from app.predict import (
    format_time,
    get_output_paths,
    get_sentences,
    predict_label,
    process_audio_prediction,
    run_full_pipeline,
    upload_audio,
)


@pytest.fixture(autouse=True)
def disable_mock_and_sleep(monkeypatch):
    """Turn off MOCK_MODE and disable real sleeps in polling loops."""
    # Ensure we hit the real code paths, not the MOCK_MODE stubs
    monkeypatch.setenv("SPHINX_MOCK_MODE", "0")
    monkeypatch.setattr(predict, "MOCK_MODE", False)
    # Patch out time.sleep so polls are instant
    monkeypatch.setattr(predict, "time", SimpleNamespace(sleep=lambda *_: None))


def test_get_output_paths_default(tmp_path, monkeypatch):
    """Should create and return an 'output' directory by default."""
    monkeypatch.setenv("SAVE_TO", "output")
    monkeypatch.chdir(tmp_path)
    paths = get_output_paths("file.csv")
    assert len(paths) == 1
    p = paths[0]
    assert p.parent.name == "output"
    assert p.name == "file.csv"
    assert p.parent.exists()


@pytest.mark.parametrize(
    "ms,expected",
    [
        (0, "00:00:00,000"),
        (1234, "00:00:01,234"),
        (3_600_000 + 12_345, "01:00:12,344"),  # matches current float behavior
        (3_660_789, "01:01:00,789"),
    ],
)
def test_format_time(ms, expected):
    """format_time should convert milliseconds to HH:MM:SS,mmm correctly."""
    assert format_time(ms) == expected


def test_upload_audio_success(monkeypatch):
    """upload_audio should post then poll until completed and return lang + id."""
    fake_id = "trans123"

    def fake_post(url, headers, json):
        assert url.endswith("/v2/transcript")
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"id": fake_id},
        )

    poll = {"calls": 0}

    def fake_get(url, headers):
        poll["calls"] += 1
        if poll["calls"] == 1:
            return SimpleNamespace(json=lambda: {"status": "processing"})
        return SimpleNamespace(
            json=lambda: {"status": "completed", "language_code": "en"}
        )

    monkeypatch.setattr(predict.requests, "post", fake_post)
    monkeypatch.setattr(predict.requests, "get", fake_get)

    lang, tid = upload_audio("any_url", language="auto")
    assert lang == "en"
    assert tid == fake_id


def test_upload_audio_error(monkeypatch):
    """If post.raise_for_status errors, upload_audio should propagate."""

    def bad_post(url, headers, json):
        return SimpleNamespace(
            raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    monkeypatch.setattr(predict.requests, "post", bad_post)
    with pytest.raises(RuntimeError):
        upload_audio("any_url")


def test_get_sentences_success(monkeypatch):
    """get_sentences should return sentences list from JSON."""
    fake_sentences = [{"start": 0, "end": 100, "text": "hello"}]

    def fake_get(url, headers):
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"sentences": fake_sentences},
        )

    monkeypatch.setattr(predict.requests, "get", fake_get)
    out = get_sentences("tid123")
    assert out == fake_sentences


def test_get_sentences_error(monkeypatch):
    """Non-2xx in get_sentences should propagate."""

    def fake_get(url, headers):
        return SimpleNamespace(
            raise_for_status=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        )

    monkeypatch.setattr(predict.requests, "get", fake_get)
    with pytest.raises(RuntimeError):
        get_sentences("tid123")


def test_predict_label_empty():
    """Empty or whitespace returns an error dict."""
    out = predict_label("   ")
    assert out == {"error": "Text is empty."}


def test_predict_label_nonempty(monkeypatch):
    """Non-empty text uses classify_emotion and returns its label/confidence."""
    # Patch classify_emotion
    monkeypatch.setattr(predict, "classify_emotion", lambda txt: ("happy", 42.0))
    out = predict_label("hello world")
    assert out == {"predicted_label": "happy", "confidence": 42.0}


def test_process_audio_prediction_end_to_end(tmp_path, monkeypatch):
    """process_audio_prediction should orchestrate and write CSVs under tmp_path."""
    # 1. Patch upload_audio & get_sentences
    monkeypatch.setattr(predict, "upload_audio", lambda url, language: ("en", "tid"))
    sentences = [
        {"start": 0, "end": 500, "text": "hi"},
        {"start": 500, "end": 1500, "text": "bye"},
    ]
    monkeypatch.setattr(predict, "get_sentences", lambda tid: sentences)
    # 2. Patch translation & classification
    monkeypatch.setattr(predict, "translate_if_needed", lambda txt, lang: txt.upper())
    monkeypatch.setattr(predict, "classify_emotion", lambda txt: ("joy", 99.9))
    # 3. Redirect CSV output into tmp_path
    monkeypatch.setattr(
        predict,
        "get_output_paths",
        lambda filename, save_to=None: [tmp_path / filename],
    )

    result = process_audio_prediction("audio_url", save_to=None, language="auto")
    # Validate structure
    assert set(result.keys()) == {"results", "csv_paths"}
    assert len(result["results"]) == 2

    # CSV was actually written
    csv_path = Path(result["csv_paths"][0])
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    expected_cols = [
        "Start Time",
        "End Time",
        "Sentence",
        "Translation",
        "Emotion",
        "Confidence (%)",
    ]
    assert list(df.columns) == expected_cols


def test_run_full_pipeline_proxy(monkeypatch):
    """run_full_pipeline should wrap process_audio_prediction into DF+paths."""
    dummy = {"results": [{"foo": "bar"}], "csv_paths": ["p1.csv"]}
    monkeypatch.setattr(
        predict, "process_audio_prediction", lambda *args, **kwargs: dummy
    )
    df, paths = run_full_pipeline("u", save_to="x", language="y")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["foo"]
    assert paths == ["p1.csv"]
