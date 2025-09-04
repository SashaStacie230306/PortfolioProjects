"""Evaluate a fine-tuned transformer model on test data for emotion classification.

This script:
- Loads the most recent test CSV file
- Encodes labels using the same label encoder from training
- Performs predictions
- Computes evaluation metrics (accuracy and F1-score)
- Saves the results to a JSON file

Args (via CLI):
    --test_data_path (str): Folder containing the test CSV(s)
    --model_path (str): Path to the trained model (inc tokenizer and label_encoder.pkl)
    --metrics_output (str): Folder where evaluation metrics will be saved
"""

import argparse
import glob
import json
import os
import pickle

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(description="Evaluate a trained model on test data.")
parser.add_argument(
    "--test_data_path", type=str, required=True, help="Directory containing test CSV(s)"
)
parser.add_argument(
    "--model_path", type=str, required=True, help="Path to the trained model directory"
)
parser.add_argument(
    "--metrics_output", type=str, required=True, help="Directory to save metrics.json"
)
args = parser.parse_args()


def find_latest_csv(folder_path):
    """Finds the most recently modified CSV file in the given folder.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        str: Full path to the latest CSV file.

    Raises:
        FileNotFoundError: If no CSV files are found in the folder.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return max(csv_files, key=os.path.getmtime)


# Load test data
test_csv_path = find_latest_csv(args.test_data_path)
df = pd.read_csv(test_csv_path)
df = df.rename(columns={"Emotion": "label"})  # Normalize column name
sentences = list(df["Sentence"])

# Load label encoder from model path
label_encoder_path = os.path.join(args.model_path, "label_encoder.pkl")
if not os.path.exists(label_encoder_path):
    raise FileNotFoundError(f"Label encoder not found at: {label_encoder_path}")

with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

true_labels = label_encoder.transform(df["label"])

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
model.eval()

# Tokenize and make predictions
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average="weighted")

# Save metrics to output directory
os.makedirs(args.metrics_output, exist_ok=True)
with open(os.path.join(args.metrics_output, "metrics.json"), "w") as f:
    json.dump({"accuracy": accuracy, "f1_score": f1}, f, indent=2)

print("Evaluation complete. Metrics saved.")
