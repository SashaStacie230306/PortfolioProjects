"""Unit tests for the app.inference module."""

import os
import sys
from types import SimpleNamespace
from typing import Any

import torch

# Setup for importing the module under test
os.environ["SPHINX_MOCK_MODE"] = "0"
if "app.inference" in sys.modules:
    del sys.modules["app.inference"]

import app.inference as inf  # noqa: E402


def test_classify_emotion_stub(monkeypatch: Any) -> None:
    """Test classify_emotion when in MOCK_MODE."""
    monkeypatch.setattr(inf, "MOCK_MODE", True)
    label, conf = inf.classify_emotion("hi")
    assert label == "neutral"
    assert conf == 0.0


def test_classify_emotion_real(monkeypatch: Any) -> None:
    """Test classify_emotion with mocked components in real mode."""

    class DummyBatch(dict):
        def to(self, device: torch.device) -> "DummyBatch":
            return self

    class DummyModel:
        def eval(self) -> None:
            pass

        def __call__(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(logits=torch.tensor([[0.1, 0.9]]))

    dummy_inputs = DummyBatch({"input_ids": torch.tensor([[1, 2, 3]])})

    def dummy_tokenizer(text, **kwargs):
        """Mock tokenizer that returns dummy input IDs."""
        return dummy_inputs

    dummy_model = DummyModel()
    dummy_encoder = SimpleNamespace(inverse_transform=lambda x: ["joy"])

    monkeypatch.setattr(inf, "MOCK_MODE", False)
    monkeypatch.setattr(inf, "tokenizer", dummy_tokenizer)
    monkeypatch.setattr(inf, "model", dummy_model)
    monkeypatch.setattr(inf, "label_encoder", dummy_encoder)
    monkeypatch.setattr(inf, "device", torch.device("cpu"))

    label, conf = inf.classify_emotion("hello world")
    assert label == "joy"
    assert isinstance(conf, float)


def test_translate_if_needed_mock(monkeypatch: Any) -> None:
    """Should return input text unchanged in MOCK_MODE."""
    monkeypatch.setattr(inf, "MOCK_MODE", True)
    assert inf.translate_if_needed("cześć", "pl") == "cześć"


def test_translate_if_needed_en(monkeypatch: Any) -> None:
    """Should return original text for non-'pl' languages."""
    monkeypatch.setattr(inf, "MOCK_MODE", False)
    assert inf.translate_if_needed("hello", "en") == "hello"


def test_translate_if_needed_translation(monkeypatch: Any) -> None:
    """Should invoke translator if lang_code is 'pl' and not in MOCK_MODE."""
    monkeypatch.setattr(inf, "MOCK_MODE", False)
    monkeypatch.setattr(
        inf,
        "translator",
        lambda text: [{"translation_text": "hi"}],
    )
    assert inf.translate_if_needed("cześć", "pl") == "hi"
