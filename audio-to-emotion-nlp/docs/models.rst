Model CLI Utilities (`src/models`)
==================================

This module provides command-line interfaces (CLI) for training, evaluating, and managing models.

Model Training CLI
------------------

.. automodule:: src.models.CLI_models
   :members:
   :undoc-members:
   :show-inheritance:

Allows you to fine-tune transformer models using custom datasets.

Example:

.. code-block:: bash

    python src/models/CLI_models.py \
        --train_path data/train/final_output.csv \
        --epochs 5 \
        --model distilbert-base-uncased

Evaluation CLI
--------------

.. automodule:: src.models.CLI_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Evaluate trained models and print classification metrics.

::

    src/
    └── models/
        ├── CLI_models.py
        ├── CLI_evaluation.py
        ├── label_encoder.pkl
        └── __init__.py
