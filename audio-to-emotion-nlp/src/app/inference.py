"""
inference.py
------------
Contains core inference logic for emotion classification and optional translation.
"""

import os

import torch

# Set MOCK_MODE for Sphinx or testing environments to skip heavy model loading
MOCK_MODE = os.getenv("SPHINX_MOCK_MODE") == "1"

if not MOCK_MODE:
    from transformers import pipeline

    from app.model_loader import get_model_components

    model, tokenizer, label_encoder = get_model_components()
    model.eval()

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Translation pipeline for Polish → English
    translator = pipeline(
        "translation",
        model="Helsinki-NLP/opus-mt-pl-en",
        tokenizer="Helsinki-NLP/opus-mt-pl-en",
        device=0 if torch.cuda.is_available() else -1,
    )

else:
    model = tokenizer = label_encoder = translator = device = None


def classify_emotion(text: str) -> tuple[str, float]:
    """Classify the emotion of a single input sentence.

    Args:
        text (str): Input sentence in English.

    Returns:
        tuple[str, float]: Predicted emotion label and its confidence score as a percentage.
    """
    if MOCK_MODE:
        return "neutral", 0.0  # stub output for docs

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        pred = torch.argmax(probs).item()

        label = label_encoder.inverse_transform([pred])[0]
        confidence = round(probs[pred].item() * 100, 2)

        return label, confidence


def translate_if_needed(text: str, lang_code: str) -> str:
    """Translate Polish text to English if the language code is 'pl'.

    Args:
        text (str): The input text in Polish or English.
        lang_code (str): The detected language code (e.g., 'pl', 'en').

    Returns:
        str: Translated English text if input was in Polish; otherwise, returns original text.
    """
    if MOCK_MODE or lang_code != "pl":
        return text
    return translator(text)[0]["translation_text"]
