CLI Utilities (`src/cli`)
==========================

This module provides the main CLI interface for running the pipeline from the command line using `argparse` and `typer`.

Typer CLI
---------

.. automodule:: src.cli.CLI_main_typer
   :members:
   :undoc-members:
   :show-inheritance:

Modern CLI tool using [Typer](https://typer.tiangolo.com/):

- `--text` for direct input
- `--file` for batch processing
- `--lang` for source language (e.g., "pl")

Classic CLI
-----------

.. automodule:: src.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

A legacy CLI using `argparse`, intended for scripted usage.

Legacy Prototype
----------------

.. automodule:: src.cli.Old_main
   :members:
   :undoc-members:
   :show-inheritance:

Deprecated version retained for comparison/testing only.

::

    src/
    └── cli/
        ├── CLI_main.py
        ├── CLI_main_typer.py
        ├── main.py
        └── Old_main.py
