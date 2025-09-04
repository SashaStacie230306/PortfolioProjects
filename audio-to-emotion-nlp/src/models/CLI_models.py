"""CLI Models for Emotion Classification."""

import pickle

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_model(checkpoint_path: str, label_encoder_path: str):
    """Load a pretrained emotion classification model, tokenizer, and label encoder.

    Args:
        checkpoint_path (str): Path to the directory containing the model checkpoint.
        label_encoder_path (str): Path to the saved label encoder (.pkl file).

    Returns:
        tuple: A tuple containing:
            - model: The loaded PyTorch transformer model for sequence classification.
            - tokenizer: The corresponding tokenizer.
            - label_encoder: A fitted sklearn LabelEncoder for decoding predicted
              labels.
            - device: The torch device (CPU or CUDA) the model is loaded onto.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(
        device
    )
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
    model.eval()
    return model, tokenizer, label_encoder, device


def classify_emotions(texts: list, model, tokenizer, label_encoder, device) -> list:
    """Predict emotion labels for a list of input sentences using a pretrained model.

    Args:
        texts (list of str): Input sentences to classify.
        model: The loaded transformer model for emotion classification.
        tokenizer: The tokenizer used to preprocess text.
        label_encoder: The LabelEncoder used to decode prediction indices into labels.
        device: The device on which the model runs (CPU or CUDA).

    Returns:
        list of str: Predicted emotion labels corresponding to the input sentences.
    """
    emotions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(
            device
        )
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=1).item()
            label = label_encoder.inverse_transform([pred])[0]
        emotions.append(label)
    return emotions
