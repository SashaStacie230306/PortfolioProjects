# Audio-to-Emotion NLP Pipeline

## Overview

This project implements a production-ready, modular NLP pipeline that processes English `.mp3` audio files and outputs structured CSV files with timestamped transcriptions and predicted emotion labels. It integrates AssemblyAI for speech-to-text and uses a fine-tuned Hugging Face transformer model for emotion classification. The codebase supports both CLI usage and deployment via a FastAPI backend.

## Features

- Transcription of English audio using AssemblyAI
- Timestamped sentence segmentation
- Emotion classification using a fine-tuned transformer model
- Scalable and modular architecture
- CLI tools and REST API support via FastAPI
- Compatible with Python 3.10

## Input

- English `.mp3` audio files
- Pretrained model checkpoint (`checkpoint-3906/`)
- Label encoder (`label_encoder.pkl`)

## Output

- A CSV file with the following columns:
  - `Start Time` – Timestamp (hh:mm:ss,ms)
  - `End Time`
  - `Sentence` – Transcribed text
  - `Emotion` – Predicted label

### Example Output

| Start Time     | End Time       | Sentence           | Emotion   |
|----------------|----------------|--------------------|-----------|
| 00:00:00,000   | 00:00:03,400   | Hi, how are you?   | happiness |

## Technologies Used

- Python 3.10
- AssemblyAI API
- Hugging Face Transformers
- PyTorch
- scikit-learn
- spaCy
- pandas, numpy, tqdm
- FastAPI (for deployment)

## Installation

### Prerequisites

- Python 3.10
- pip or Poetry

### Setup Using Poetry (Recommended)

```bash
git clone https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp7
cd nlp7

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

poetry install
```
### PyTorch Installation (Manual)

PyTorch is not included in the default environment to ensure macOS compatibility.
Install the correct version for your platform from: https://pytorch.org/get-started/locally/

macOS(CPU)
```bash
bashCopyEditpoetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
Windows/Linux
```bash
bashCopyEditpoetry run pip install torch torchvision torchaudio
```