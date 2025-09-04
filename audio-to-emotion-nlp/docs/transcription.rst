Transcription (`src/transcription`)
====================================

This module handles the transcription of audio and video input using **AssemblyAI**, enabling multimodal emotion classification.

Transcription CLI
-----------------

.. automodule:: src.transcription.CLI_transcribe
   :members:
   :undoc-members:
   :show-inheritance:

Command-line tool for sending audio files to AssemblyAI and retrieving transcribed text.

Example:

.. code-block:: bash

    python src/transcription/CLI_transcribe.py \
        --input data/mp3/Sondy_FIXED2.mp3 \
        --language pl

Transcription Engine
--------------------

.. automodule:: src.transcription.transcribe
   :members:
   :undoc-members:
   :show-inheritance:

Core functions:
- Upload to AssemblyAI
- Monitor job status
- Translate (if required)
- Return cleaned text for classification

AssemblyAI API Key
------------------

Make sure `.env` includes:

.. code-block:: text

    ASSEMBLYAI_API_KEY=your_key_here

::

    src/
    └── transcription/
        ├── CLI_transcribe.py
        ├── transcribe.py
        └── __init__.py
