import json
import logging
import os

import joblib
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """Initialize the model, tokenizer, and label encoder for emotion classification.

    Loads all necessary components from the Azure ML model mount directory.
    """
    global model, tokenizer, label_encoder

    # Get model directory path
    base_path = os.environ.get("AZUREML_MODEL_DIR", ".")
    try:
        subdirs = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
        artifact_dir = os.path.join(base_path, subdirs[0])
    except Exception as e:
        raise RuntimeError(f"Could not find model directory under {base_path}: {e}")

    model_path = os.path.join(artifact_dir, "model.safetensors")
    encoder_path = os.path.join(artifact_dir, "label_encoder.pkl")

    logger.info(f"Loading tokenizer and model from {artifact_dir}")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=7
    )
    model.load_state_dict(load_file(model_path))
    model.eval()

    logger.info(f"Loading label encoder from {encoder_path}")
    label_encoder = joblib.load(encoder_path)
    logger.info("Model, tokenizer, and label encoder loaded successfully.")


def run(raw_data):
    """Perform emotion prediction from a JSON input string.

    Args:
        raw_data (str): A JSON-formatted string containing a "text" field.

    Returns:
        dict: Dictionary with keys:
            - "text" (str): The input text.
            - "predicted_label" (str): Predicted emotion label.
            - "confidence" (float): Confidence score of the prediction.
            - or "error" (str): Error message if input is invalid.
    """
    logger.info(f"Received request data: {raw_data}")
    data = json.loads(raw_data)
    text = data.get("text", "").strip()
    if not text:
        logger.warning("Empty input received.")
        return {"error": "Empty input"}

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).numpy().flatten()
        idx = int(np.argmax(probs))
        pred = label_encoder.inverse_transform([idx])[0]

    logger.info(f"Prediction: {pred} (confidence {probs[idx]:.4f})")
    return {
        "text": text,
        "predicted_label": pred,
        "confidence": float(probs[idx]),
    }
