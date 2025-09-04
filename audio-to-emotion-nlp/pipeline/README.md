# Emotion Classification Pipeline using Azure ML

This project implements an **end-to-end machine learning pipeline** for fine-tuning a transformer-based model to classify emotions in text. The pipeline is orchestrated using **Azure Machine Learning** and includes components for training, evaluation, and conditional registration of the model based on performance.

---

## Project Structure

```
project-root/
│
├── pipeline/
│   ├── training_pipeline.py     # Main Azure ML pipeline definition and submission
│   └──README.md
│
├── src/
│   └── comp/
│       ├── train.py             # Fine-tunes the model on labeled data
│       ├── evaluate.py          # Evaluates model and saves metrics
│       ├──register.py           # Registers the model if F1 score improves
        └──config.json           # Azure credentials and workspace config
```

---

## 🔧 Components

### 1. `train.py`
- Loads training and validation CSVs.
- Fine-tunes a transformer model using Hugging Face.
- Saves model, tokenizer, label encoder, and training metadata.

### 2. `evaluate.py`
- Loads the latest test CSV.
- Predicts emotion labels using the fine-tuned model.
- Computes **accuracy** and **F1-score**.
- Saves results to `metrics.json`.

### 3. `register.py`
- Compares the new F1 score with the best version of the registered model.
- Registers the new model to Azure ML **only if performance improves**.

---

## Azure ML Pipeline: `training_pipeline.py`

This script defines and submits an Azure ML pipeline consisting of three steps:

### ➤ **Train Step**
```bash
python train.py \
  --train_data_path <path> \
  --val_data_path <path> \
  --base_model_path <path> \
  --model_output_path <output>
```
- Uses a Hugging Face model as base.
- Saves the fine-tuned model to a designated output folder.

### ➤ **Evaluate Step**
```bash
python evaluate.py \
  --test_data_path <path> \
  --model_path <model> \
  --metrics_output <output>
```
- Runs inference on test set.
- Outputs evaluation metrics.

### ➤ **Register Step**
```bash
python register.py \
  --model <path> \
  --metrics <path>
```
- Registers the model in Azure ML if the **F1 score is higher** than the current best.

---

## Running the Pipeline

To submit the pipeline:

```bash
python pipeline/training_pipeline.py
```

This script will:
- Load Azure credentials from `config.json`
- Use the **latest registered model** as a base
- Train and evaluate on specified dataset URIs
- Register the model if performance improves

You can track the job's status via the URL printed in the console.

---

## ✅ Requirements

- Azure ML SDK (`azure-ai-ml`)
- `transformers`, `torch`, `scikit-learn`, `pandas`
- Configured `config.json` with Azure credentials:

```json
{
  "tenant_id": "...",
  "client_id": "...",
  "client_secret": "...",
  "subscription_id": "...",
  "resource_group": "...",
  "workspace_name": "..."
}
```

---

## Notes

- All components use a shared environment defined in Azure ML (`nlp7-emotion-env`, version `32`).
- All steps run on **serverless compute**.
- The base model is retrieved via MLflow model registry using the `latest` tag.

---

## Output Artifacts
- Fine-tuned model and tokenizer
- `label_encoder.pkl`
- `metrics.json`
- Registered model in Azure ML (if F1 improves)
