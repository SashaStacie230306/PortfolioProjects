Utilities (`src/utils`)
========================

These modules contain shared functionality for logging, formatting, and general utilities across the NLP7 pipeline.

Helper Functions
----------------

.. automodule:: src.utils.helpers
   :members:
   :undoc-members:
   :show-inheritance:

Includes functions like:

- `load_label_encoder(path)`
- `clean_text(text)`
- `translate_polish_to_english(text)`

Logger
------

.. automodule:: src.utils.logger
   :members:
   :undoc-members:
   :show-inheritance:

Wraps Python's `logging` module for consistent pipeline logs.

Logs to both console and file (`logs/app.log`) with format:
[%(asctime)s] %(levelname)s - %(name)s - %(message)s


::

    src/
    └── utils/
        ├── helpers.py
        ├── logger.py
        └── __init__.py

