"""
model_loader.py
---------------
Loads the fine-tuned transformer model, tokenizer, and label encoder
used for emotion classification.
"""

import logging
import os
import pickle
from pathlib import Path

from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Enable mock mode when building documentation (e.g., with Sphinx)
MOCK_MODE = os.getenv("SPHINX_MOCK_MODE") == "1"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not MOCK_MODE:
    # Resolve base directory of the project (fallback to ../../ if BASE_DIR not set)
    BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).resolve().parents[2]))

    # Define paths using pathlib
    MODEL_DIR = BASE_DIR / "src" / "checkpoint-3906"
    LABEL_ENCODER_PATH = BASE_DIR / "src" / "models" / "label_encoder.pkl"

    # Validate paths
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

    logger.info(f"Loading model from: {MODEL_DIR}")
    logger.info(f"Loading label encoder from: {LABEL_ENCODER_PATH}")

    # Load model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)

    # Load label encoder
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
else:
    model = tokenizer = label_encoder = None


def get_model_components() -> (
    tuple[RobertaForSequenceClassification, RobertaTokenizer, object]
):
    """Return the fine-tuned model, tokenizer, and label encoder.

    Returns:
        tuple: A tuple containing:
            - model (RobertaForSequenceClassification): The fine-tuned transformer model.
            - tokenizer (RobertaTokenizer): Tokenizer for the model.
            - label_encoder (object): Fitted label encoder for decoding prediction indices.
    """
    return model, tokenizer, label_encoder
