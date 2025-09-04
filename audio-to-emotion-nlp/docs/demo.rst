Demo Interface (`src/demo`)
===========================

This module provides an interactive demo interface for emotion classification using **Gradio**, as well as a Python client for REST API calls.

Gradio UI
---------

.. automodule:: src.demo.app_ui
   :members:
   :undoc-members:
   :show-inheritance:

Launches a local UI where users can test predictions on text/audio:

.. code-block:: bash

    python src/demo/app_ui.py

Supports:
- Text input
- Audio (MP3/WAV) upload
- Optional transcription with AssemblyAI

API Client
----------

.. automodule:: src.demo.api_client
   :members:
   :undoc-members:
   :show-inheritance:

A lightweight client for sending text/audio to the `/predict/` FastAPI endpoint.

Gradio Assets
-------------

Includes:
- `certificate.pem` (for HTTPS)
- `requirements.txt` for UI deployment

::

    src/
    └── demo/
        ├── app_ui.py
        ├── api_client.py
        ├── .gradio/
        └── demo_README.md
