NLP7 Emotion Classification Pipeline
====================================

Introduction
------------

Welcome to the documentation for the NLP7 multimodal emotion classification system, developed as part of an applied research project. This pipeline is designed to identify emotional states from text, audio, and video inputs using transformer-based models.

The system includes SenseAI, a front-end application that enables users to interact with the models in real time. It provides clear, accessible emotion predictions across different input types via an intuitive user interface built with Gradio.

Key Features
---------------

- SenseAI interactive frontend app (Gradio)
- FastAPI-powered REST API for emotion inference
- CLI tools for model interaction, testing, and preprocessing
- Docker and Azure integration with AKS (Kubernetes)
- Speech-to-text transcription via AssemblyAI
- Cloud-first and local-first architecture
- Configurable with `.env` and `environment.yaml`
- Modular architecture under `src/`
- Pytest-driven test coverage

----

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Contents:

   app
   cli
   config
   data
   demo
   deployment
   emotion
   inference
   model_loader
   models
   predict
   transcription
   utils
   testing
   emotion

----

Project Resources
-----------------

- **GitHub**: https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp7/
- **Environment setup**: ``environment.yaml``
- **Deployment instructions**: See `deployment.rst`
- **Cloud strategy & cost evaluation**: `project_plan/Cloud_Cost_Evaluation.docx`


Getting Started
===================

Installation
------------

To set up the NLP7 project locally:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp7.git
      cd nlp7

2. Create and activate the environment:

   .. code-block:: bash

      conda env create -f environment.yaml
      conda activate nlp-pipeline

3. Optional: Enable Sphinx mock mode for documentation:

   .. code-block:: bash

      export SPHINX_MOCK_MODE=1

4. Set up `.env` for API keys and endpoints:
   - See `.env` template in the project root for variable references used across the pipeline and Azure integration.

Usage Overview
--------------

- Launch FastAPI backend:

  .. code-block:: bash

     uvicorn src.app.API_main:app --reload

- Run CLI inference:

  .. code-block:: bash

     python src/cli/main.py --text "I feel great today!"

- Preprocess datasets:

  .. code-block:: bash

     python src/data/CLI_data_preprocessing.py

- Transcribe an audio file:

  .. code-block:: bash

     python src/transcription/CLI_transcribe.py --input path/to/audio.mp3

- Launch the SenseAI demo UI:

  .. code-block:: bash

     python src/demo/app_ui.py

----

Cloud Deployment (Azure)
============================

Azure deployment scripts are located in `src/deployment/`:

- `deploy_kubernetes_endpoint.py`: Create Azure endpoint (if not exists)
- `cloud_deploy.py`: Deploy latest model version
- `score.py`: Custom scoring script (used in deployment)

Requirements:

- Azure credentials via CLI or managed identity
- Azure ML model and environment registered
- Kubernetes inference cluster provisioned

----

Testing
==========

Tests are located under the `tests/` directory.

To run tests:

.. code-block:: bash

   pytest

----

Contributors
===============

- Sasha Stacie  
- Monika Stangenberg  
- Deuza Borges Varela  
- Kamil Łęga  
- Celine Wu

----

Indices and Tables
======================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
