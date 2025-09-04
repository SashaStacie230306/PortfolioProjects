Data Module (`src/data`)
=========================

Handles preprocessing of raw datasets and audio files into model-ready formats for training and inference.

Data Preprocessing CLI
----------------------

.. automodule:: src.data.CLI_data_preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

CLI utility for preprocessing `.csv`, `.mp3`, or `.txt` input into labeled output for training or prediction.

Example:

.. code-block:: bash

    python src/data/CLI_data_preprocessing.py \
        --input data/rest/test.txt \
        --output data/processed/output.csv

Data Ingestion
--------------

.. automodule:: src.data.data_ingestion
   :members:
   :undoc-members:
   :show-inheritance:

Ingest and validate raw data from multiple modalities (text/audio) into structured CSVs.

File Layout
-----------

::

    src/
    └── data/
        ├── CLI_data_preprocessing.py
        ├── data_ingestion.py
        ├── mp3/
        ├── processed/
        ├── train/
        ├── rest/
        └── __init__.py
