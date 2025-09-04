"""Evaluate a trained emotion classification model on a labeled test dataset."""

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_test_set(model, tokenizer, df_test, label_encoder, device):
    """Evaluate a trained emotion classification model on a labeled test dataset.

    The function computes accuracy, precision, recall, and F1 score by:
    - Tokenizing test sentences.
    - Running inference with the model.
    - Comparing predictions to true labels.

    Args:
        model: A pretrained transformer model for sequence classification.
        tokenizer: The tokenizer corresponding to the model.
        df_test (pd.DataFrame): DataFrame containing the test data. Must include
            columns "Sentence" and "Emotion_Label".
        label_encoder: Fitted LabelEncoder used to encode emotion labels.
        device: Torch device to run inference on (e.g., 'cpu' or 'cuda').

    Returns:
        Tuple[dict, np.ndarray]:
            - A dictionary with evaluation metrics: accuracy, precision, recall, and
              F1 score.
            - A NumPy array of predicted label indices.
    """
    encodings = tokenizer(
        df_test["Sentence"].tolist(), truncation=True, padding=True, return_tensors="pt"
    )
    encodings = {key: val.to(device) for key, val in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()

    true_labels = label_encoder.transform(df_test["Emotion_Label"])

    metrics = {
        "accuracy": accuracy_score(true_labels, preds),
        "f1": f1_score(true_labels, preds, average="weighted"),
        "precision": precision_score(true_labels, preds, average="weighted"),
        "recall": recall_score(true_labels, preds, average="weighted"),
    }

    return metrics, preds
