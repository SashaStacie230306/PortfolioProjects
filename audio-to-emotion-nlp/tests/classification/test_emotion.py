"""Test the `classify_emotions` function and its import-time logic."""

import builtins
import importlib
import io
import os
import pickle
import sys
import types
from typing import Any, Dict, List

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def reload_emotion_module(mock_mode: bool):
    """Reload src.classification.emotion under a given MOCK_MODE.

    Args:
        mock_mode (bool): If True, sets SPHINX_MOCK_MODE='1' before import;
                          otherwise unsets it.

    Returns:
        module: The reloaded emotion module.
    """
    if mock_mode:
        os.environ["SPHINX_MOCK_MODE"] = "1"
    else:
        os.environ.pop("SPHINX_MOCK_MODE", None)
    return importlib.reload(importlib.import_module("src.classification.emotion"))


def test_import_time_real_branch(tmp_path, monkeypatch) -> None:
    """When imported with MOCK_MODE unset, module‐level loading runs without error.

    This test:
      1) Injects a dummy pandas module so helpers.py import succeeds.
      2) Stubs builtins.open for each label_encoder.pkl load to return a fresh BytesIO.
      3) Stubs transformers and torch.device to avoid external I/O.
      4) Reloads emotion.py in real mode and checks its module‐level vars.
    """
    # 1) Prevent real pandas
    dummy_pd = types.ModuleType("pandas")
    dummy_pd.DataFrame = type("DataFrame", (), {})
    monkeypatch.setitem(sys.modules, "pandas", dummy_pd)

    # 2) Real‐mode import
    monkeypatch.delenv("SPHINX_MOCK_MODE", raising=False)
    monkeypatch.chdir(tmp_path)

    # 3) Stub open() for label_encoder.pkl
    dummy_encoder = "DUMMY_ENCODER"
    real_open = builtins.open

    def open_only_label_encoder(path, mode="rb", *args, **kwargs):
        if str(path).endswith("label_encoder.pkl"):
            return io.BytesIO(pickle.dumps(dummy_encoder))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_only_label_encoder)

    # 4) Stub transformers + torch.device
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda name: "TOK_STUB")

    class DummyMod:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, device: Any) -> "DummyMod":
            return self

        def eval(self) -> None:
            pass

        def __call__(self, **kw) -> Any:
            class R:
                logits = torch.tensor([[0.0, 0.0, 0.0]])

            return R()

    monkeypatch.setattr(
        AutoModelForSequenceClassification,
        "from_pretrained",
        lambda ckpt: DummyMod(),
    )
    monkeypatch.setattr(torch, "device", lambda spec: "cpu-dev")

    # 5) Reload in real mode
    emo = reload_emotion_module(mock_mode=False)

    # 6) Check module‐level assignments
    assert emo.label_encoder == dummy_encoder
    assert emo.tokenizer == "TOK_STUB"
    assert isinstance(emo.model, DummyMod)
    assert emo.device == "cpu-dev"


def test_classify_emotions_in_mock_mode() -> None:
    """MOCK_MODE=True at import should return ['neutral']*n."""
    emo = reload_emotion_module(mock_mode=True)
    assert emo.classify_emotions(["a", "b", "c"]) == ["neutral", "neutral", "neutral"]


def test_classify_emotions_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """With runtime MOCK_MODE off, stubbed internals return correct labels."""
    emo = reload_emotion_module(mock_mode=True)
    emo.MOCK_MODE = False
    emo.device = "cpu"

    # Stub tokenizer
    class DummyInputs(dict):
        def __init__(self) -> None:
            super().__init__(
                {
                    "input_ids": torch.tensor([[1]]),
                    "attention_mask": torch.tensor([[1]]),
                }
            )

        def to(self, device: Any) -> "DummyInputs":
            return self

    monkeypatch.setattr(emo, "tokenizer", lambda txt, **kw: DummyInputs())

    # Stub model
    class StubModel:
        def __call__(self, **kwargs: Any):
            class R:
                logits = torch.tensor([[0.0, 0.0, 0.5]])

            return R()

    monkeypatch.setattr(emo, "model", StubModel())

    # Stub label encoder
    class StubEncoder:
        def inverse_transform(self, arr: List[int]) -> List[str]:
            return ["my_label"]

    monkeypatch.setattr(emo, "label_encoder", StubEncoder())

    # Capture logger calls
    calls: Dict[str, str] = {}

    class DummyLogger:
        def info(self, msg, *args):
            calls["info"] = msg % args if args else msg

        def error(self, msg, *args):
            calls["error"] = msg % args if args else msg

    monkeypatch.setattr(emo, "logger", DummyLogger())

    out = emo.classify_emotions(["hello"])
    assert out == ["my_label"]
    assert "Emotion classification completed for 1 sentences." in calls["info"]


def test_classify_emotions_error_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """If tokenizer raises, returns ['unknown']*n and logs an error."""
    emo = reload_emotion_module(mock_mode=True)
    emo.MOCK_MODE = False
    emo.device = "cpu"

    # Force tokenizer to raise
    monkeypatch.setattr(
        emo,
        "tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    # Capture logger calls
    calls: Dict[str, str] = {}

    class DummyLogger:
        def info(self, msg, *args):
            calls.setdefault("info", msg % args if args else msg)

        def error(self, msg, *args):
            calls["error"] = msg % args if args else msg

    monkeypatch.setattr(emo, "logger", DummyLogger())

    out = emo.classify_emotions(["x", "y"])
    assert out == ["unknown", "unknown"]
    assert "Error during emotion classification" in calls["error"]


def test_classify_emotions_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty list input should return [] and log completion."""
    emo = reload_emotion_module(mock_mode=True)
    emo.MOCK_MODE = False
    emo.device = "cpu"

    # No tokenizer/model activity; will skip to final logging
    monkeypatch.setattr(
        emo,
        "tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no call")),
    )
    monkeypatch.setattr(emo, "model", object())
    monkeypatch.setattr(emo, "label_encoder", object())

    # Capture logger calls
    calls: Dict[str, str] = {}

    class DummyLogger:
        def info(self, msg, *args):
            calls["info"] = msg % args if args else msg

        def error(self, msg, *args):
            calls.setdefault("error", msg % args if args else msg)

    monkeypatch.setattr(emo, "logger", DummyLogger())

    out = emo.classify_emotions([])
    assert out == []
    assert "Emotion classification completed for 0 sentences." in calls["info"]
