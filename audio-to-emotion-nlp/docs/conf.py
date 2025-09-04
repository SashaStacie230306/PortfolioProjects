# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath('../src'))

# Ensure the environment variable is set for mock imports
os.environ["SPHINX_MOCK_MODE"] = "1"

# Add the root and src directories to sys.path so autodoc can find modules
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'NLP Pipeline'
copyright = '2025, Sasha Stacie, Monika Stangenberg, Deuza Borges Varela, Kamil Łęga, Celine Wu'
author = 'Sasha Stacie, Monika Stangenberg, Deuza Borges Varela, Kamil Łęga, Celine Wu'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_mock_imports = [
    "torch",
    "transformers",
    "sklearn",
    "pandas",
    "requests",
    "yt_dlp",
    "fastapi",
    "uvicorn",
    "dotenv",
    "assemblyai",   
    "ffmpeg",     
]

# Options for HTML output
html_theme = 'sphinx_rtd_theme'


