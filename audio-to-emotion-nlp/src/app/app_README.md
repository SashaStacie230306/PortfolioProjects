
# App — Core Logic for Emotion Classification API

This folder contains the primary logic for handling multimodal emotion classification, including inference from text, audio, and video. It supports integration with Azure ML for text classification and uses AssemblyAI for transcription. All components are orchestrated via the FastAPI server defined in `API_main.py`.

---

## File Descriptions

### `predict.py`
Handles the complete prediction pipeline for audio and video files:
- Uploads audio to AssemblyAI and polls for transcription results
- Retrieves sentence-level timestamps and content
- Translates from Polish to English if needed
- Performs emotion classification per sentence
- Outputs structured results as CSV

### `inference.py`
Contains the emotion classification logic for text:
- Loads the transformer model (RoBERTa) and tokenizer
- Applies a label encoder to map predictions to emotion labels
- Provides translation from Polish to English using the `Helsinki-NLP/opus-mt-pl-en` model

### `model_loader.py`
Loads and initializes model components:
- `RobertaForSequenceClassification` from a local `checkpoint-XXXX/` directory
- Corresponding tokenizer
- Pickled label encoder (`label_encoder.pkl`) for decoding class indices

> These files are only required **if running local inference**. When using Azure ML, these local assets are **not needed**.

### `azure_client.py`
Handles communication with an Azure ML online endpoint:
- Reads endpoint details from `.env`
- Sends POST requests with raw text input for emotion classification
- Parses and returns predicted label and confidence

---

## How It Works

1. **Media Input (audio/video):**
   - `predict.py` uploads the file to AssemblyAI → transcribes it
   - Transcripts are processed sentence-by-sentence
   - Translations (Polish → English) are done if needed
   - Emotions are predicted using Azure or (optionally) local inference
   - Results are saved using Pandas in `final_predictions.csv`

2. **Text Input:**
   - Routed via `API_main.py`
   - Calls `azure_client.py` to send the text to Azure ML for classification
   - Saves the result to `text_prediction.csv`

---

## Environment Variables (`.env` Required)

To run this module properly, define the following in a `.env` file:

```env
# AssemblyAI for transcription
ASSEMBLYAI_API_KEY=your_key_here

# Azure ML Endpoint for text inference
AZURE_ENDPOINT_URL=https://your-endpoint-url
AZURE_API_KEY=your_azure_key_here

# Output control
SAVE_TO=output,downloads

# Optional: Disable real model/API calls (for testing or Sphinx docs)
SPHINX_MOCK_MODE=1
```

---

## Mock Mode

When `SPHINX_MOCK_MODE=1`, all heavyweight tasks like:
- Model loading
- Transcription
- API calls to Azure/AssemblyAI

...are bypassed to support faster development or documentation generation.

---

## Used By

This logic is invoked by:
- **`API_main.py`** — FastAPI web server that defines API endpoints
- **`cloud_deploy.py`** (in `src/deployment/`) — Deploys models to Azure Kubernetes Online Endpoint using `KubernetesOnlineDeployment`

---

## Output Format

Predictions (media-based or text) are saved as CSV files:

| Column           | Description                                |
|------------------|--------------------------------------------|
| `Start Time`     | Timestamp in `HH:MM:SS,ms` format          |
| `Sentence`       | Sentence from transcript                   |
| `Translation`    | English version (if translated from Polish)|
| `Emotion`        | Predicted label (e.g. happy, angry)        |
| `Confidence (%)` | Confidence score in percentage             |

---

## Notes

- No local model files (`checkpoint-XXXX/`, `label_encoder.pkl`) are required when using Azure ML for inference.
- Only needed if you run offline inference locally (e.g., for testing or research).
- CSV output location is defined by `SAVE_TO` in your `.env` (can be `output/`, `downloads`, `desktop`, etc.)

---

## Cleanup Utility

Temporary uploads (in `temp_uploads/`) are cleaned automatically after 30 minutes of inactivity using the `clean_temp_uploads()` function in `API_main.py`.

---

## Dependencies

All dependencies are defined in [`environment.yaml`](../environment.yaml).
To set up the environment using Conda:

```bash
conda env create -f environment.yaml
conda activate nlp7
```

---

## Authors

- Sasha Stacie
- Monika Stangenberg
- Deuza Borges Varela
- Kamil Łęga
- Celine Wu

---
