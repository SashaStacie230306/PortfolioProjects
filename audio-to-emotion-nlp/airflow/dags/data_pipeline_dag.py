"""Airflow DAG to submit a transcription and preprocessing job to Azure ML.

This DAG pulls credentials and environment variables from an Airflow connection and uses
Azure ML to execute a script that transcribes an audio file, cleans text, and uploads
the result to blob storage.
"""

from datetime import datetime

from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from azure.ai.ml import MLClient, command
from azure.identity import ClientSecretCredential

from airflow import DAG

# === Configuration ===
CONFIG = {
    "env_name": "nlp7-emotion-env:32",
    "script_path": "./scripts",
    "script_file": "transcribe_and_split.py",
    "compute": "serverless",
    "experiment": "transcribe_preprocess_pipeline",
    "job_desc": "Transcribe audio, clean text and upload",
    "audio_path": (
        "azureml://datastores/workspaceblobstore/paths/"
        "UI/2025-06-15_125636_UTC/raw_data/Inside Polands Communist City  NOWA HUTA.mp3"
    ),
    "output_blob": "Users/235065/csv/final_output.csv",
}


def get_ml_client():
    """Authenticate and return Azure ML client using Airflow connection."""
    conn = BaseHook.get_connection("azure_ml_conn")
    extras = conn.extra_dejson
    credential = ClientSecretCredential(
        tenant_id=extras["tenant_id"],
        client_id=extras["client_id"],
        client_secret=extras["client_secret"],
    )
    return (
        MLClient(
            credential,
            subscription_id=extras["subscription_id"],
            resource_group_name=extras["resource_group"],
            workspace_name=extras["workspace_name"],
        ),
        extras["assemblyai_api_key"],
        extras["azure_storage_connection_string"],
    )


def run_transcription_pipeline():
    """Submit and stream the Azure ML job with environment variables from Airflow."""
    ml_client, assemblyai_api_key, storage_conn_str = get_ml_client()

    job = command(
        code=CONFIG["script_path"],
        command=f"python {CONFIG['script_file']}",
        environment=f"azureml:{CONFIG['env_name']}",
        compute=CONFIG["compute"],
        experiment_name=CONFIG["experiment"],
        description=CONFIG["job_desc"],
        environment_variables={
            "ASSEMBLYAI_API_KEY": assemblyai_api_key,
            "AZURE_STORAGE_CONNECTION_STRING": storage_conn_str,
            "AUDIO_PATH": CONFIG["audio_path"],
            "OUTPUT_BLOB_PATH": CONFIG["output_blob"],
        },
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(submitted_job.name)
    return submitted_job.name


# === DAG Declaration ===
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="nlp7_transcribe_preprocess_job",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["azureml", "assemblyai", "preprocessing"],
) as dag:

    run_transcribe_preprocess_pipeline = PythonOperator(
        task_id="run_transcribe_preprocess_pipeline",
        python_callable=run_transcription_pipeline,
    )
