"""Upload and register the final_output.csv file as a data asset in Azure ML.

workspace.
"""

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from azureml.core import Datastore, Workspace

# Base directory is one level up from this script (i.e., project root)
base_dir = Path(__file__).resolve().parent.parent

# Define paths relative to project structure
config_path = base_dir / "config.json"
csv_path = base_dir / "src" / "data" / "processed" / "final_output.csv"

# Step 1: Load Azure ML Workspace from config.json
ml_client = MLClient.from_config(
    credential=DefaultAzureCredential(),
    path=str(config_path),
)

workspace = Workspace.from_config(path=str(config_path))

# Step 2: Access the default datastore (usually 'workspaceblobstore')
datastore = Datastore.get(workspace, datastore_name="workspaceblobstore")

# Step 3: Upload the CSV file to the target path in the datastore
datastore.upload_files(
    files=[str(csv_path)],
    target_path="Users/235065/csv",
    overwrite=True,
    show_progress=True,
)

# Step 4: Register the uploaded file as a URI_FILE data asset
data_asset = Data(
    name="final_output_csv",
    version="1",
    type=AssetTypes.URI_FILE,
    path=(
        "azureml://datastores/workspaceblobstore/paths/"
        "Users/235065/csv/final_output.csv"
    ),
    description="Final processed CSV file",
)

# Step 5: Register (or update) the data asset in Azure ML
ml_client.data.create_or_update(data_asset)
