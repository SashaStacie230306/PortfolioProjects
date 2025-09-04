"""This script defines an Azure ML pipeline for training and evaluating an emotion.

classification model.
"""

import json
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

# Load workspace configuration
with open("config.json") as f:
    config = json.load(f)

# Connect to Azure ML
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=config["subscription_id"],
    resource_group_name=config["resource_group"],
    workspace_name=config["workspace_name"],
)

# Environment & path
environment_name = "nlp7-emotion-env"
environment_version = "32"
component_path = str(Path(__file__).resolve().parent.parent / "src" / "comp")

# Load environment
env = ml_client.environments.get(environment_name, environment_version)


# Get the latest version of the model
def get_latest_model_uri():
    """Get the latest version URI of the Emotion-classification model."""
    try:
        latest_model = ml_client.models.get("Emotion-classification", label="latest")
        model_uri = f"azureml:{latest_model.name}:{latest_model.version}"
        print(f"Using latest model: {model_uri}")
        return model_uri
    except Exception as e:
        print(f"Error getting latest model: {e}")
        return "azureml:Emotion-classification:1"


# Training component
train_component = command(
    name="train",
    display_name="Train model",
    inputs={
        "train_data_path": Input(type="uri_folder"),
        "val_data_path": Input(type="uri_folder"),
        "base_model": Input(type="mlflow_model"),
    },
    outputs={"model_output_path": Output(type="uri_folder", mode="rw_mount")},
    code=component_path,
    command=(
        "python train.py "
        "--train_data_path ${{inputs.train_data_path}} "
        "--val_data_path ${{inputs.val_data_path}} "
        "--base_model_path ${{inputs.base_model}} "
        "--model_output_path ${{outputs.model_output_path}}"
    ),
    environment=env,
)

# Evaluation component
evaluate_component = command(
    name="evaluate",
    display_name="Evaluate model",
    inputs={
        "test_data_path": Input(type="uri_folder"),
        "model_path": Input(type="uri_folder"),
    },
    outputs={
        "metrics_output": Output(type="uri_folder", mode="rw_mount"),
    },
    code=component_path,
    command=(
        "python evaluate.py "
        "--test_data_path ${{inputs.test_data_path}} "
        "--model_path ${{inputs.model_path}} "
        "--metrics_output ${{outputs.metrics_output}}"
    ),
    environment=env,
)

# Register component
register_component = command(
    name="register",
    display_name="Register model if improved",
    inputs={
        "model": Input(type="uri_folder"),
        "metrics": Input(type="uri_folder"),
    },
    code=component_path,
    command=(
        "python register.py "
        "--model ${{inputs.model}} "
        "--metrics ${{inputs.metrics}}"
    ),
    environment=env,
)


# Define the pipeline
@pipeline(name="emotion-pipeline")
def emotion_pipeline(train_data, val_data, test_data, base_model):
    """Azure ML pipeline for training, evaluating, and registering an emotion.

    classification model.
    """
    train_step = train_component(
        train_data_path=train_data, val_data_path=val_data, base_model=base_model
    )
    train_step.compute = "serverless"

    eval_step = evaluate_component(
        test_data_path=test_data, model_path=train_step.outputs.model_output_path
    )
    eval_step.compute = "serverless"

    register_step = register_component(
        model=train_step.outputs.model_output_path,
        metrics=eval_step.outputs.metrics_output,
    )
    register_step.compute = "serverless"


# Submit the pipeline
if __name__ == "__main__":
    latest_model_uri = get_latest_model_uri()

    job = emotion_pipeline(
        train_data=Input(
            type="uri_folder",
            path=(
                "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
                "resourcegroups/buas-y2/workspaces/NLP7-2025/"
                "datastores/workspaceblobstore/paths/UI/2025-06-17_093618_UTC/"
            ),
        ),
        val_data=Input(
            type="uri_folder",
            path=(
                "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
                "resourcegroups/buas-y2/workspaces/NLP7-2025/"
                "datastores/workspaceblobstore/paths/UI/2025-06-17_093441_UTC/"
            ),
        ),
        test_data=Input(
            type="uri_folder",
            path=(
                "azureml://subscriptions/0a94de80-6d3b-49f2-b3e9-ec5818862801/"
                "resourcegroups/buas-y2/workspaces/NLP7-2025/"
                "datastores/workspaceblobstore/paths/UI/2025-06-12_230903_UTC/"
            ),
        ),
        base_model=Input(type="mlflow_model", path=latest_model_uri),
    )

    job = ml_client.jobs.create_or_update(job, experiment_name="emotion-experiment")
    print(f"Pipeline submitted! Track here:\n{job.studio_url}")
