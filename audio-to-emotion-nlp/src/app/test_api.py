"""Module to test API endpoint at http://localhost:8000/."""

import pytest
import requests

pytest.skip("Skipping API tests — not ready yet", allow_module_level=True)


response = requests.get("http://localhost:8000/")
data = response.json()

print(data)
