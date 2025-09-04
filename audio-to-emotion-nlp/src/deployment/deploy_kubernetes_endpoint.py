from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    KubernetesOnlineDeployment,
    KubernetesOnlineEndpoint,
    OnlineRequestSettings,
)
from azure.identity import DefaultAzureCredential

CANDIDATE_MODELS = ["Emotion-classification-full", "Emotion-classification"]
ENDPOINT_NAME = "emotion-endpoint-v2"
DEPLOYMENT_NAME = "blue"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group_name="buas-y2",
    workspace_name="NLP7-2025",
)

# Ensure endpoint exists
try:
    ml_client.online_endpoints.get(ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' exists.")
except:
    print(f"Creating endpoint: {ENDPOINT_NAME}")
    ep = KubernetesOnlineEndpoint(
        name=ENDPOINT_NAME, auth_mode="key", compute="adsai-lambda-0"
    )
    ml_client.begin_create_or_update(ep).result()
    print("Endpoint created.")


# Pick the latest model version
all_models = []
for name in CANDIDATE_MODELS:
    all_models.extend(ml_client.models.list(name=name))
if not all_models:
    raise RuntimeError(f"No models found in {CANDIDATE_MODELS}")
latest = sorted(all_models, key=lambda m: int(m.version), reverse=True)[0]
print(f"Using latest model: {latest.name} v{latest.version}")

# Get the inference environment
envs = list(ml_client.environments.list(name="nlp7-inference-env"))
if not envs:
    raise RuntimeError("Environment 'nlp7-inference-env' not found.")
env = envs[0]

# 4) Define the deployment (no custom CPU/memory)
deployment = KubernetesOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=latest,
    environment=env,
    code_configuration=CodeConfiguration(
        code="src/deployment", scoring_script="score.py"
    ),
    instance_type="gpu",
    instance_count=1,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=120000,  # 2-minute timeout
        max_concurrent_requests_per_instance=1,  # limit for memory usage
        max_queue_wait_ms=60000,  # 1-minute queue wait
    ),
)

# Update (or create) in place
print(f"Updating deployment '{DEPLOYMENT_NAME}' → model v{latest.version}")
ml_client.begin_create_or_update(deployment).result()
print(f"Deployment '{DEPLOYMENT_NAME}' is up‐to‐date.")

# (Re)affirm 100% traffic
endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
endpoint.traffic = {DEPLOYMENT_NAME: 100}
ml_client.begin_create_or_update(endpoint).result()
print("Traffic remains at 100% on 'blue'.")
print(f"Endpoint '{ENDPOINT_NAME}' now serving {latest.name} v{latest.version}")
