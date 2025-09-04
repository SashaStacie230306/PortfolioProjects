import os
import time
from io import StringIO
import requests

import nltk
import pandas as pd
from nltk.corpus import stopwords
from azureml.core import Workspace, Datastore
from azure.storage.blob import BlobServiceClient

nltk.download("stopwords")


def remove_stopwords(input_text):
    """Remove English stopwords from a given input text."""
    words = input_text.split()
    clean_words = [word for word in words if word.lower() not in stopwords.words("english")]
    return " ".join(clean_words)


def upload_file(filepath: str, headers: dict) -> str:
    """Upload a file to AssemblyAI and return the upload URL."""
    with open(filepath, "rb") as f:
        response = requests.post(
            "https://api.assemblyai.com/v2/upload",
            headers=headers,
            files={"file": f}
        )
    response.raise_for_status()
    return response.json()["upload_url"]


def request_transcription(upload_url: str, headers: dict) -> str:
    """Request transcription from AssemblyAI and wait until completion."""
    endpoint = "https://api.assemblyai.com/v2/transcript"
    payload = {
        "audio_url": upload_url,
        "language_code": "en",
        "punctuate": True,
        "format_text": True
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    transcript_id = response.json()["id"]

    polling_endpoint = f"{endpoint}/{transcript_id}"
    while True:
        poll = requests.get(polling_endpoint, headers=headers).json()
        if poll["status"] == "completed":
            break
        elif poll["status"] == "error":
            raise Exception(f"Transcription failed: {poll['error']}")
        time.sleep(5)

    return transcript_id


def format_time(ms: int) -> str:
    """Format milliseconds into SRT timestamp format."""
    seconds = ms / 1000
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def get_transcript_df(transcript_id: str, headers: dict) -> pd.DataFrame:
    """Retrieve sentence-level transcript from AssemblyAI and return a DataFrame."""
    url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}/sentences"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    sentences = response.json()["sentences"]

    return pd.DataFrame([{
        "Start Time": format_time(s["start"]),
        "End Time": format_time(s["end"]),
        "Sentence": s["text"]
    } for s in sentences])


def transcribe_audio(file_path: str, headers: dict) -> pd.DataFrame:
    """Run full transcription pipeline on audio file."""
    upload_url = upload_file(file_path, headers)
    transcript_id = request_transcription(upload_url, headers)
    return get_transcript_df(transcript_id, headers)


def main():
    """Main entrypoint for transcription and preprocessing script."""
    # Load API key
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise EnvironmentError("ASSEMBLYAI_API_KEY environment variable not set.")
    headers = {"authorization": api_key}

    # Initialize workspace and blob client
    workspace = Workspace.get(
        name="NLP7-2025",
        resource_group="buas-y2",
        subscription_id="0a94de80-6d3b-49f2-b3e9-ec5818862801"
    )
    datastore = Datastore.get(workspace, "workspaceblobstore")
    blob_service_client = BlobServiceClient.from_connection_string(datastore.account_key)
    container_name = datastore.container_name

    # Define blob paths
    blob_input_path = "UI/2025-06-23_144945_UTC/raw_data/Inside Polands Communist City  NOWA HUTA.mp3"
    blob_output_path = "UI/2025-06-19_123719_UTC/processed_data/transcribed_output.csv"
    local_audio_file = "input_audio.mp3"

    # Download input
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_input_path)
    with open(local_audio_file, "wb") as f:
        f.write(blob_client.download_blob().readall())

    # Transcribe and clean
    df = transcribe_audio(local_audio_file, headers)
    df["Clean Sentence"] = df["Sentence"].apply(remove_stopwords)

    # Upload output
    buffer = StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    output_client = blob_service_client.get_blob_client(container=container_name, blob=blob_output_path)
    output_client.upload_blob(buffer.read(), overwrite=True)

    print("✅ Transcription + preprocessing complete.")


if __name__ == "__main__":
    main()
