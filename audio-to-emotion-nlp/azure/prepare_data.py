import os
import pandas as pd
from azure.identity import ClientSecretCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# === Credentials ===
credential = ClientSecretCredential(
    tenant_id="0a33589b-0036-4fe8-a829-3ed0926af886",
    client_id="a2230f31-0fda-428d-8c5c-ec79e91a49f5",
    client_secret="AWA8Q~14jhEuWoP5K4FNnRfsRc_Qcbhx8PeLRaXw"
)

# === Connect to Azure ML workspace ===
ml_client = MLClient(
    credential=credential,
    subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801",
    resource_group_name="buas-y2",
    workspace_name="NLP7-2025"
)

# === Auto-increment version helper ===
def get_next_version(ml_client, asset_name):
    existing_versions = ml_client.data.list(name=asset_name)
    version_numbers = []
    for asset in existing_versions:
        try:
            version_numbers.append(int(asset.version))
        except ValueError:
            continue
    return str(max(version_numbers, default=0) + 1)

# === Preprocess CSV ===
csv_path = "./final_predictions.csv"
df = pd.read_csv(csv_path)

# 1. Remove null rows
df.dropna(inplace=True)

# 2. Trim whitespace in text columns
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# 3. Optionally remove stopwords from 'text' column
stopwords = {
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "at", "with", "without",
    "is", "are", "was", "were", "this", "that", "i", "me", "we", "you", "he", "she", "it", "they"
}
if 'text' in df.columns:
    df['text'] = df['text'].astype(str).apply(
        lambda s: ' '.join(word for word in s.split() if word.lower() not in stopwords)
    )

# Save cleaned data back to file
df.to_csv(csv_path, index=False)

# === Register cleaned data with auto-incremented version ===
asset_name = "emotion-csv-single"
version = get_next_version(ml_client, asset_name)

data_asset = Data(
    name=asset_name,
    version=version,
    path=csv_path,
    type=AssetTypes.URI_FILE,
    description="Cleaned CSV for emotion classification (auto-versioned)"
)

registered_data = ml_client.data.create_or_update(data_asset)

# === Output confirmation ===
print("✅ Uploaded and registered successfully.")
print(f"🔗 Data path: {registered_data.path}")
print(f"📦 Asset name: {registered_data.name}:{registered_data.version}")
