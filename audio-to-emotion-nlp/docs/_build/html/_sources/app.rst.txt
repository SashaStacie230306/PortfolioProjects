Backend API (`src/app`)
=========================

This module implements the **FastAPI** REST service for interacting with the NLP7 pipeline via HTTP.

Main App
--------

.. automodule:: src.app.API_main
   :members:
   :undoc-members:
   :show-inheritance:

Launch with:

.. code-block:: bash

    uvicorn src.app.API_main:app --reload

Routes defined here expose:

- `/` – Root
- `/predict/text` – Text emotion classification
- `/predict/audio` – Audio emotion classification
- `/health` – Readiness checks

Model Loader
------------

.. automodule:: src.app.model_loader
   :members:
   :undoc-members:
   :show-inheritance:

Loads trained transformer models and the label encoder from `checkpoint-XXXX`.

Prediction Logic
----------------

.. automodule:: src.app.predict
   :members:
   :undoc-members:
   :show-inheritance:

Performs inference on user-submitted inputs.

Inference Engine
----------------

.. automodule:: src.app.inference
   :members:
   :undoc-members:
   :show-inheritance:

Routes text/audio through model pipeline.

Azure Client (Optional)
-----------------------

.. automodule:: src.app.azure_client
   :members:
   :undoc-members:
   :show-inheritance:

Optional cloud routing module to send requests to deployed AzureML endpoints.

::

    src/
    └── app/
        ├── API_main.py
        ├── inference.py
        ├── model_loader.py
        ├── predict.py
        ├── azure_client.py

         