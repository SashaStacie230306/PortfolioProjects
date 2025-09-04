"""Train a transformer-based model for emotion classification.

This script:
- Loads training and validation datasets from CSV files
- Fine-tunes a Hugging Face model using PyTorch
- Saves the model, tokenizer, label encoder, and training metadata

Args (via CLI):
    --train_data_path (str): Path to training data folder (with .csv files)
    --val_data_path (str): Path to validation data folder
    --base_model_path (str): Path to base model checkpoint (e.g., from MLFlow or
                             Hugging Face)
    --model_output_path (str): Directory where the trained model and artifacts will
                               be saved
"""

import argparse
import glob
import json
import os
import pickle

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser(
    description="Train a transformer-based emotion classifier."
)
parser.add_argument(
    "--train_data_path",
    type=str,
    required=True,
    help="Directory containing training CSV(s)",
)
parser.add_argument(
    "--val_data_path",
    type=str,
    required=True,
    help="Directory containing validation CSV(s)",
)
parser.add_argument(
    "--base_model_path",
    type=str,
    required=True,
    help="Path to base transformer model directory",
)
parser.add_argument(
    "--model_output_path",
    type=str,
    required=True,
    help="Directory to save the trained model",
)
args = parser.parse_args()


def find_latest_csv(folder_path):
    """Finds the most recently modified CSV file in a given directory.

    Args:
        folder_path (str): Path to a directory containing .csv files.

    Returns:
        str: Full path to the latest .csv file.

    Raises:
        FileNotFoundError: If no .csv files are found in the directory.
    """
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return max(csv_files, key=os.path.getmtime)


class EmotionDataset(Dataset):
    """A PyTorch Dataset for emotion classification using a Hugging Face tokenizer.

    Attributes:
        sentences (List[str]): List of input sentences.
        labels (List[int]): Encoded labels.
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding text.
    """

    def __init__(self, df, tokenizer, label_encoder):
        """Initializes the dataset.

        Args:
            df (pd.DataFrame): DataFrame with 'Sentence' and 'label' columns.
            tokenizer (PreTrainedTokenizer): Hugging Face tokenizer.
            label_encoder (LabelEncoder): Fitted LabelEncoder.
        """
        self.sentences = df["Sentence"].tolist()
        self.labels = label_encoder.transform(df["label"].tolist())
        self.tokenizer = tokenizer

    def __len__(self):
        """Returns: int: Number of samples in the dataset."""
        return len(self.sentences)

    def __getitem__(self, idx):
        """Tokenizes and returns one data sample.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary with 'input_ids', 'attention_mask', and 'labels' tensors.
        """
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# Load tokenizer and model
print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(args.base_model_path)
print("Model loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load CSV files
train_csv = find_latest_csv(args.train_data_path)
val_csv = find_latest_csv(args.val_data_path)

# Load and rename columns
train_df = pd.read_csv(train_csv).rename(columns={"Emotion": "label"})
val_df = pd.read_csv(val_csv).rename(columns={"Emotion": "label"})

# Fit label encoder
label_encoder = LabelEncoder()
label_encoder.fit(train_df["label"])

# Prepare datasets and dataloaders
train_dataset = EmotionDataset(train_df, tokenizer, label_encoder)
val_dataset = EmotionDataset(val_df, tokenizer, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Set up training
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = CrossEntropyLoss()
epochs = 3

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} training loss: {total_loss / len(train_loader):.4f}")

# Save model and supporting files
os.makedirs(args.model_output_path, exist_ok=True)
model.save_pretrained(args.model_output_path)
tokenizer.save_pretrained(args.model_output_path)

# Save label encoder
with open(os.path.join(args.model_output_path, "label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Save training metadata
info = {
    "base_model": args.base_model_path,
    "labels": label_encoder.classes_.tolist(),
    "num_labels": len(label_encoder.classes_),
}
with open(os.path.join(args.model_output_path, "training_info.json"), "w") as f:
    json.dump(info, f, indent=2)

print("Training complete. Model saved to:", args.model_output_path)
