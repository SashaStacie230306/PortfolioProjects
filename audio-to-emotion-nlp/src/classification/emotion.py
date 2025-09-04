"""Emotion classification using a fine-tuned transformer model."""

import os
import pickle
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.logger import get_logger

# module-scoped logger
logger = get_logger(__name__)

# Set MOCK_MODE for Sphinx or lightweight environments
MOCK_MODE = os.getenv("SPHINX_MOCK_MODE") == "1"

# Load the model and tokenizer
if not MOCK_MODE:
    # Setup paths
    checkpoint_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "checkpoint-3906")
    )
    label_encoder_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "label_encoder.pkl"
    )

    # Load label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
else:
    # Dummy values for Sphinx compatibility
    model = tokenizer = label_encoder = device = None


def classify_emotions(texts: List[str]) -> List[str]:
    """Classify the emotion of each sentence using a pretrained transformer model.

    Args:
        texts (list of str): A list of input sentences to classify.

    Returns:
        list of str: Predicted emotion labels corresponding to each sentence.
    """
    if MOCK_MODE:
        return ["neutral"] * len(texts)  # Stub response

    emotions = []
    try:
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, padding=True
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=1).item()
                label = label_encoder.inverse_transform([pred])[0]
            emotions.append(label)
        logger.info("Emotion classification completed for %d sentences.", len(texts))
        return emotions
    except Exception as e:
        logger.error("Error during emotion classification: %s", e)
        return ["unknown"] * len(texts)
