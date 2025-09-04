from datetime import datetime

from airflow.hooks.base import BaseHook
from airflow.operators.python import PythonOperator
from azure.ai.ml import Input, MLClient
from azure.ai.ml.entities import CommandJob, Data
from azure.identity import ClientSecretCredential

from airflow import DAG

# ========== Configuration ==========
CONFIG = {
    "data_asset_name": "data_raw",
    "splits": ["train", "val", "test"],
    "env_name": "nlp7-emotion-env:32",
    "script_path": "./split_script",  # relative to the DAGs folder
    "script_file": "split.py",
}


# ========== Utility Functions ==========
def get_azure_credentials():
    conn = BaseHook.get_connection("azure_ml_conn")
    extras = conn.extra_dejson
    return (
        ClientSecretCredential(
            tenant_id=extras["tenant_id"],
            client_id=extras["client_id"],
            client_secret=extras["client_secret"],
        ),
        extras,
    )


def get_ml_client():
    credential, extras = get_azure_credentials()
    return MLClient(
        credential,
        subscription_id=extras["subscription_id"],
        resource_group_name=extras["resource_group"],
        workspace_name=extras["workspace_name"],
    )


def create_command_job(ml_client, data_asset):
    return CommandJob(
        code=CONFIG["script_path"],
        command=f"python {CONFIG['script_file']} --input_data ${{{{inputs.input_data}}}}",
        inputs={"input_data": Input(path=data_asset.id, type="uri_folder")},
        environment=f"azureml:{CONFIG['env_name']}",
        compute="adsai-lambda-0",
        experiment_name="split_pipeline",
    )


def register_data_output(ml_client, job_name, split):
    output_path = (
        f"azureml://datastores/workspaceblobstore/paths/{job_name}/outputs/{split}.csv"
    )
    asset = Data(
        name=f"{split}_folder", path=output_path, type="uri_file", version=None
    )
    ml_client.data.create_or_update(asset)


# ========== DAG Tasks ==========
def trigger_split_job():
    ml_client = get_ml_client()
    data_asset = ml_client.data.get(name=CONFIG["data_asset_name"], label="latest")
    job = create_command_job(ml_client, data_asset)
    submitted = ml_client.jobs.create_or_update(job)
    ml_client.jobs.stream(submitted.name)
    return submitted.name


def register_outputs(ti):
    ml_client = get_ml_client()
    job_name = ti.xcom_pull(task_ids="trigger_split_job")
    for split in CONFIG["splits"]:
        register_data_output(ml_client, job_name, split)


# ========== DAG Definition ==========
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="cloud_split_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["azureml", "versioned", "uri_folder"],
) as dag:

    trigger = PythonOperator(
        task_id="trigger_split_job", python_callable=trigger_split_job
    )

    register = PythonOperator(
        task_id="register_outputs", python_callable=register_outputs
    )

    trigger >> register
