import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    KubernetesOnlineDeployment,
    CodeConfiguration,
    OnlineRequestSettings,
)

# Configuration
SUBSCRIPTION_ID = "0a94de80-6d3b-49f2-b3e9-ec5818862801"
RESOURCE_GROUP = "buas-y2"
WORKSPACE_NAME = "NLP7-2025"
ENDPOINT_NAME = "emotion-endpoint-v2"
ENVIRONMENT_NAME = "nlp7-inference-env"
MODEL_NAME = "Emotion-classification"

# Resolve absolute path to current file location
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Authenticate
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Get latest version of the target model
models = list(ml_client.models.list(name=MODEL_NAME))
if not models:
    raise RuntimeError(f"No model found with name: {MODEL_NAME}")

# Sort by version (descending) and pick the latest
def safe_version(m):
    return int(m.version) if m.version.isdigit() else -1

latest_model = sorted(models, key=safe_version, reverse=True)[0]
print(f"Using model: {latest_model.name} (version {latest_model.version})")

# Get environment
env = ml_client.environments.get(name=ENVIRONMENT_NAME, version="1")

# Deploy both blue and green
for DEPLOYMENT_NAME in ["blue", "green"]:
    print(f"Deploying model to Azure endpoint as '{DEPLOYMENT_NAME}'...")
    deployment = KubernetesOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=latest_model,
        environment=env,
        instance_type="defaultinstancetype",
        code_configuration=CodeConfiguration(
            code=CODE_DIR,
            scoring_script="score.py"
        ),
        instance_count=1,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=120000,
            max_concurrent_requests_per_instance=1,
            max_queue_wait_ms=60000,
        ),
    )
    ml_client.begin_create_or_update(deployment).result()
    print(f"Deployment '{DEPLOYMENT_NAME}' completed.")

# Route traffic: 80% to blue, 20% to green
endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
endpoint.traffic = {"blue": 80, "green": 20}
ml_client.begin_create_or_update(endpoint).result()

# Confirm details
online_endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint URL: {online_endpoint.scoring_uri}")
print(f"Deployed model: {latest_model.name} v{latest_model.version}")
print("Traffic split:")
for name, pct in endpoint.traffic.items():
    print(f" - {name}: {pct}%")
