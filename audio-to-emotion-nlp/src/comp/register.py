"""Register a model in Azure ML it's F1 score is higher than the best existing version.

This script:
- Loads evaluation metrics (F1 score) for the newly trained model
- Compares to the best existing version (if any)
- Registers the new model as the latest version if it's better, or if no F1 exists yet

Args (via CLI):
    --model (str): Path to the trained model directory
    --metrics (str): Path to directory containing metrics.json
    --config (str): Path to config.json for Azure credentials (optional)
"""

import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import ClientSecretCredential

# -----------------------------
# Parse input arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Register new model if F1 improves.")
parser.add_argument(
    "--model", type=str, required=True, help="Path to the model directory"
)
parser.add_argument(
    "--metrics", type=str, required=True, help="Path to the directory with metrics.json"
)
parser.add_argument(
    "--config", type=str, default="config.json", help="Path to credentials config file"
)
args = parser.parse_args()

# -----------------------------
# Load credentials from config
# -----------------------------
with open(args.config) as f:
    config = json.load(f)

credential = ClientSecretCredential(
    tenant_id=config["tenant_id"],
    client_id=config["client_id"],
    client_secret=config["client_secret"],
)

ml_client = MLClient(
    credential=credential,
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)

# -----------------------------
# Load new model metrics
# -----------------------------
with open(os.path.join(args.metrics, "metrics.json")) as f:
    metrics = json.load(f)

new_f1 = float(metrics.get("f1_score", 0.0))
model_name = "Emotion-classification"

# -----------------------------
# Find best existing F1 score
# -----------------------------
best_f1 = 0.0
for m in ml_client.models.list(name=model_name):
    if "f1_score" in m.tags:
        try:
            best_f1 = max(best_f1, float(m.tags["f1_score"]))
        except ValueError:
            continue

# -----------------------------
# Register model if F1 improves
# -----------------------------
if new_f1 > best_f1 or best_f1 == 0.0:
    print(f"✅ Registering model: new F1 = {new_f1}, best F1 = {best_f1}")

    model = Model(
        path=args.model,
        name=model_name,
        description="Emotion classifier (auto-registered)",
        tags={"f1_score": str(new_f1)},
    )

    registered = ml_client.models.create_or_update(model)
    print(f"Model registered as version {registered.version} and labeled 'latest'.")
else:
    print(f"ℹModel not registered: new F1 ({new_f1}) <= best F1 ({best_f1})")
