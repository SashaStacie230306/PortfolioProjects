Testing (`tests/`)
===================

The NLP7 pipeline includes a full test suite written with `pytest`.

Test Organization
-----------------

::

    tests/
    ├── app/
    ├── classification/
    ├── cli/
    ├── config/
    ├── data/
    ├── train/
    ├── transcription/
    ├── utils/
    └── conftest.py

Tests cover:
- API endpoints
- Inference logic
- Logger and helper utilities
- Emotion classification functions
- Azure cloud clients

Examples
--------

Run all tests:

.. code-block:: bash

    pytest tests/

Run specific module:

.. code-block:: bash

    pytest tests/app/test_predict.py

Configuration
-------------

- `pytest.ini` includes base markers
- `conftest.py` holds shared fixtures
- Coverage is supported via `.coveragerc`

::

    ├── .coveragerc
    ├── pytest.ini
    └── tests/
