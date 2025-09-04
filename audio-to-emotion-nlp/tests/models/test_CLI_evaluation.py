"""Unit tests for src/models/CLI_evaluation.py."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder

from src.models.CLI_evaluation import evaluate_test_set


class DummyModel:
    """Simulate a HuggingFace model with a logits attribute."""

    def __init__(self, logits_tensor):
        """Initialize with a tensor of logits."""
        self._logits = logits_tensor

    def __call__(self, **kwargs):
        """Simulate model call that returns logits."""
        # Return an object with a .logits attribute
        return type("R", (), {"logits": self._logits})


class DummyTokenizer:
    """Simulate a tokenizer that returns dict of tensors."""

    def __init__(self):
        """Initialize with no called arguments."""
        self.called_with = None

    def __call__(self, texts, truncation, padding, return_tensors):
        """Simulate tokenization by storing called arguments."""
        self.called_with = {
            "texts": texts,
            "truncation": truncation,
            "padding": padding,
            "return_tensors": return_tensors,
        }
        # Create dummy tensors: shape (len(texts), 5) just as example
        batch_size = len(texts)
        return {
            "input_ids": torch.zeros(batch_size, 5, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
        }


@pytest.fixture
def simple_test_df():
    """Create a simple test DataFrame with two sentences and labels."""
    return pd.DataFrame(
        {
            "Sentence": ["foo", "bar"],
            "Emotion_Label": ["happy", "sad"],
        }
    )


@pytest.fixture
def label_encoder():
    """Create a LabelEncoder for 'happy' and 'sad' labels."""
    le = LabelEncoder()
    le.fit(["happy", "sad"])
    return le


def test_perfect_prediction(simple_test_df, label_encoder):
    """If logits predict the true label exactly, metrics should all be 1.0."""
    # true labels are [0,1] because label_encoder.transform maps 'happy'->0, 'sad'->1
    # Create logits that ensure argmax matches true labels
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    model = DummyModel(logits)
    tokenizer = DummyTokenizer()

    # Run evaluation on 'cpu'
    metrics, preds = evaluate_test_set(
        model=model,
        tokenizer=tokenizer,
        df_test=simple_test_df,
        label_encoder=label_encoder,
        device="cpu",
    )

    # All metrics should be perfect
    assert pytest.approx(metrics["accuracy"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["precision"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["recall"], rel=1e-6) == 1.0
    assert pytest.approx(metrics["f1"], rel=1e-6) == 1.0

    # Predictions array should match [0,1]
    assert isinstance(preds, np.ndarray)
    assert np.array_equal(preds, np.array([0, 1]))

    # Tokenizer should have been called with the right args
    assert tokenizer.called_with["texts"] == ["foo", "bar"]
    assert tokenizer.called_with["truncation"] is True
    assert tokenizer.called_with["padding"] is True
    assert tokenizer.called_with["return_tensors"] == "pt"


def test_mixed_prediction(simple_test_df, label_encoder):
    """Test a mix of correct and incorrect predictions yields expected metrics."""
    # True labels: [0,1]; let's predict [1,0] so both wrong
    logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    model = DummyModel(logits)
    tokenizer = DummyTokenizer()

    metrics, preds = evaluate_test_set(
        model=model,
        tokenizer=tokenizer,
        df_test=simple_test_df,
        label_encoder=label_encoder,
        device="cpu",
    )

    # Predictions are [1,0]; true [0,1] → accuracy=0
    assert pytest.approx(metrics["accuracy"], abs=1e-6) == 0.0
    # For binary balanced labels, precision and recall also zero
    assert pytest.approx(metrics["precision"], abs=1e-6) == 0.0
    assert pytest.approx(metrics["recall"], abs=1e-6) == 0.0
    assert pytest.approx(metrics["f1"], abs=1e-6) == 0.0

    assert np.array_equal(preds, np.array([1, 0]))


def test_empty_dataframe(label_encoder):
    """An empty DataFrame should result in empty preds and metrics raising or zero."""
    df_empty = pd.DataFrame({"Sentence": [], "Emotion_Label": []})
    model = DummyModel(torch.empty((0, 2)))
    tokenizer = DummyTokenizer()

    # When no samples, accuracy_score returns 0.0 by default
    metrics, preds = evaluate_test_set(
        model=model,
        tokenizer=tokenizer,
        df_test=df_empty,
        label_encoder=label_encoder,
        device="cpu",
    )

    # No predictions
    assert preds.size == 0
    # Depending on sklearn version, metrics may be 0.0
    assert "accuracy" in metrics and isinstance(metrics["accuracy"], float)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
