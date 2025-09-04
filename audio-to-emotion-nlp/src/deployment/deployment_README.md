
# Deployment — Azure ML Kubernetes Inference

This folder (`src/deployment/`) contains all scripts and assets required to deploy the NLP emotion classification model to an **Azure Kubernetes Online Endpoint**. It includes deployment automation, scoring logic, and endpoint management.

---

## Contents

### `cloud_deploy.py`
Deploys the **latest version** of a registered model to an existing endpoint. It:
- Uses `KubernetesOnlineDeployment`
- References an existing environment and model
- Pushes scoring logic defined in `score.py`
- Routes 100% traffic to the specified deployment

Usage:
```bash
python src/deployment/cloud_deploy.py
```

### `deploy_kubernetes_endpoint.py`
Creates or updates the **KubernetesOnlineEndpoint** if it doesn't already exist. It:
- Creates the endpoint (if missing)
- Picks the most recent model version from a list of candidate model names
- Deploys it using GPU compute (`instance_type="gpu"`)
- Ensures 100% traffic is routed to the deployment

Usage:
```bash
python src/deployment/deploy_kubernetes_endpoint.py
```

### `score.py`
Defines the **inference logic** that runs on Azure once a request hits the endpoint. It:
- Loads `model.safetensors` and `label_encoder.pkl` from Azure’s model directory
- Initializes a RoBERTa model with correct labels
- Tokenizes the input and returns a prediction with confidence

This is the file referenced by `CodeConfiguration(..., scoring_script="score.py")`.

---

## Environment Expectations

These deployment scripts assume:
- A model (with `model.safetensors` and `label_encoder.pkl`) is already registered in Azure ML
- A managed Kubernetes compute is available and named (e.g., `adsai-lambda-0`)
- An environment called `nlp7-inference-env` exists in your Azure workspace
- Authentication is handled via `DefaultAzureCredential` (use Azure CLI login or managed identity)

---

## Endpoint Behavior

- The deployed endpoint accepts JSON payloads like:
```json
{
  "text": "I feel great today!"
}
```

- The response will be:
```json
{
  "text": "I feel great today!",
  "predicted_label": "joy",
  "confidence": 0.9845
}
```

---

## Notes

- `cloud_deploy.py` assumes the endpoint already exists.
- `deploy_kubernetes_endpoint.py` can both create and update endpoints.
- `score.py` **must match** the input/output schema expected by your frontend or FastAPI backend.

---

## Dependencies

These are managed via the root-level `environment.yaml`. Ensure the following packages are included:
- `azure-ai-ml`
- `transformers`
- `torch`
- `safetensors`
- `scikit-learn` or `joblib`
- `numpy`

---

## Summary

| Script                      | Purpose                               |
|-----------------------------|----------------------------------------|
| `cloud_deploy.py`           | Deploys latest model to Azure          |
| `deploy_kubernetes_endpoint.py` | Creates endpoint + deploys model    |
| `score.py`                  | Azure ML scoring logic for inference   |

This setup ensures your model is always live and production-ready on Azure's scalable infrastructure.

---
