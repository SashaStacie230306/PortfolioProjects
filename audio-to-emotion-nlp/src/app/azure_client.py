import os
import requests
from dotenv import load_dotenv

# Load .env so AZURE_ENDPOINT_URL and AZURE_API_KEY are populated
load_dotenv()

AZURE_ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

if not AZURE_ENDPOINT_URL or not AZURE_API_KEY:
    raise RuntimeError("AZURE_ENDPOINT_URL and AZURE_API_KEY must be set in your .env")


def azure_predict(text: str) -> dict:
    """Send a prediction request to a Kubernetes-deployed Azure ML endpoint.

    Args:
        text (str): The input sentence to classify.

    Returns:
        dict: { "predicted_label": "...", "confidence": 99.12 }

    Raises:
        RuntimeError: On any HTTP error, timeout, or model-side error.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}",
    }

    payload = {"text": text}

    print("[DEBUG] Sending payload to Azure:", payload)

    try:
        resp = requests.post(
            AZURE_ENDPOINT_URL, headers=headers, json=payload, timeout=15
        )
        resp.raise_for_status()
        result = resp.json()

        # If scoring script bails with {"error":"..."} on bad input,
        # surface it so FastAPI can catch & log it properly
        if "error" in result:
            raise RuntimeError(f"Model error: {result['error']}")

        return {
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
        }

    except requests.Timeout as e:
        raise RuntimeError(f"Azure prediction timed out: {e}") from e

    except requests.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        body = e.response.text if e.response else ""
        print("[DEBUG] Azure HTTP Error", status, body)
        raise RuntimeError(f"Azure returned HTTP {status}: {body}") from e

    except requests.RequestException as e:
        raise RuntimeError(f"Azure prediction failed: {e}") from e
