"""Unit tests for src/models/CLI_models.py."""

import pickle
from types import SimpleNamespace

import pytest
import torch
import transformers

from src.models.CLI_models import classify_emotions, load_model


class DummyEncoder:
    """A simple label encoder substitute that is picklable."""

    def inverse_transform(self, idxs):
        """Simulate label encoding by returning a list of labels."""
        return [f"label_{i}" for i in idxs]


class DummyModel:
    """Simulate HuggingFace model with .to(), .eval(), and callable returning logits."""

    def __init__(self, logits_tensor):
        """Initialize with a tensor of logits."""
        self._logits = logits_tensor
        self.to_called_with = None
        self.eval_called = False

    def to(self, device):
        """Simulate moving model to a device."""
        self.to_called_with = device
        return self

    def eval(self):
        """Simulate setting model to evaluation mode."""
        self.eval_called = True

    def __call__(self, **kwargs):
        """Simulate model call returning logits."""
        return SimpleNamespace(logits=self._logits)


class DummyTokenizer:
    """Simulate a tokenizer returning a dict subclass with .to()."""

    def __call__(self, text, return_tensors, truncation, padding):
        """Simulate tokenization returning a dict with input_ids and attention_mask."""

        class FakeInputs(dict):
            def to(self, device):
                return self

        return FakeInputs(
            {"input_ids": torch.tensor([[1]]), "attention_mask": torch.tensor([[1]])}
        )


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Ensure torch.cuda.is_available() is False so device is CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_load_model(tmp_path, monkeypatch):
    """load_model should load model, tokenizer, label_encoder, and device."""
    # Prepare checkpoint dir and pickled encoder file
    checkpoint_dir = tmp_path / "checkpoint_dir"
    checkpoint_dir.mkdir()
    encoder_file = tmp_path / "encoder.pkl"
    orig_encoder = DummyEncoder()
    with open(encoder_file, "wb") as f:
        pickle.dump(orig_encoder, f)

    # Prepare dummy model and tokenizer
    dummy_logits = torch.tensor([[0.1, 0.9]])
    dummy_model = DummyModel(dummy_logits)
    dummy_tokenizer = "TOK"

    # Monkey-patch HF classes
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, path: dummy_tokenizer),
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda cls, path: dummy_model),
    )

    # Call load_model
    model, tokenizer, label_encoder, device = load_model(
        str(checkpoint_dir), str(encoder_file)
    )

    # Assertions
    assert tokenizer == dummy_tokenizer
    assert model is dummy_model
    # The loaded label_encoder should be an instance of DummyEncoder
    assert isinstance(label_encoder, DummyEncoder)
    # Device is CPU
    assert device.type == "cpu"
    # model.to() and model.eval() were called
    assert model.to_called_with == device
    assert model.eval_called


def test_classify_emotions():
    """classify_emotions should tokenize each text, run model, and decode labels."""
    texts = ["a", "b", "c"]
    # logits picking index 1
    logits = torch.tensor([[0.0, 1.0, 0.0]])
    model = DummyModel(logits)
    tokenizer = DummyTokenizer()
    encoder = DummyEncoder()
    device = torch.device("cpu")

    preds = classify_emotions(texts, model, tokenizer, encoder, device)
    assert preds == ["label_1", "label_1", "label_1"]
