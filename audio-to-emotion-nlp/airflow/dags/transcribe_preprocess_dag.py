from airflow import DAG
from airflow.utils.dates import days_ago
from airflow_provider_azure_machinelearning.operators.machine_learning.job import AzureMachineLearningCreateJobOperator
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Job
from azure.ai.ml import command
import os

with DAG(
    dag_id="transcribe_preprocess_dag",
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=["azureml", "transcription"],
) as dag:

    # Set path to the script (adjust if needed)
    code_dir = os.path.join(os.path.dirname(__file__), "..", "jobs")

    # Define the Azure ML command job
    azureml_job: Job = command(
        code=code_dir,
        command="python Transcribe_process_job.py",
        environment="nlp7-emotion-env:32",       # 👈 Must exist in Azure ML
        compute="adsai-lambda-0",                # 👈 Must exist in Azure ML
        display_name="transcribe-preprocess-job",
        experiment_name="transcription_pipeline",
    )

    # Submit the job to Azure ML using the Azure Airflow provider
    submit_transcription_job = AzureMachineLearningCreateJobOperator(
        task_id="submit_transcription_job",
        conn_id="azure_ml_conn",  # 👈 Uses credentials from this connection
        job=azureml_job
    )
