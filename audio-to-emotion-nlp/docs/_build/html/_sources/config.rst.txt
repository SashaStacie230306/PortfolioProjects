Configuration (`src/config`)
============================

This module manages runtime settings such as environment variables, API keys, and deployment parameters for the NLP7 pipeline.

Configuration Module
--------------------

.. automodule:: src.config.config
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions:
- Loads `.env` variables securely via `dotenv`
- Makes configuration accessible globally
- Supports toggling between local/cloud inference

Example Usage:

.. code-block:: python

    from config import config

    print(config.AZURE_ENDPOINT_URL)

Environment Variables
---------------------

Located in the `.env` file at project root.

Structure:

.. code-block:: dotenv

    # NLP7 - Local & Azure Inference
    ASSEMBLYAI_API_KEY=your_assemblyai_key
    SAVE_TO=downloads,desktop,output

    # Azure deployment
    USE_AZURE=1
    AZURE_API_KEY=your_azure_api_key
    AZURE_ENDPOINT_URL=http://your-aks-endpoint/api/...

    # Azure workspace details
    AZURE_SUBSCRIPTION_ID=your_subscription_id
    AZURE_RESOURCE_GROUP=your_rg
    AZURE_WORKSPACE_NAME=your_workspace
    AZURE_ENDPOINT_NAME=your_endpoint_name

    # Local FastAPI access
    API_URL="http://127.0.0.1:8000"

Sensitive credentials are **never** committed to version control. They're loaded securely using `python-dotenv`.

::

    src/
    └── config/
        ├── config.py
        └── __init__.py
