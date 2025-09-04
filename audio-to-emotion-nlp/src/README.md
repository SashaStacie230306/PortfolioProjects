# Audio-to-Emotion NLP Pipeline

This project is a complete pipeline that processes Polish audio files, transcribes them using AssemblyAI, translates the sentences to English, extracts NLP features, classifies emotions using a fine-tuned transformer, and outputs the result.

## Input
- `.mp3` audio files (Polish speech)
- Trained emotion classification model (`checkpoint-3906/`)
- Label encoder (`label_encoder.pkl`)

## Output
A CSV file with the following columns:
- `Start Time` – timestamp in `hh:mm:ss,ms` format
- `End Time`
- `Sentence` – Polish sentence
- `Translation` – English version
- `Emotion` – Predicted emotion label

Example output:

| Start Time | End Time | Sentence | Translation | Emotion |
|------------|----------|----------|-------------|---------|
| 00:00:00,000 | 00:00:03,400 | Cześć, jak się masz? | Hi, how are you? | happiness |

## Features
- Real timestamps from AssemblyAI
- Translation using Helsinki-NLP's PL-EN model
- Emotion classification via transformer checkpoint
- Optional evaluation (F1, Confusion Matrix)
- Intermediate CSVs for translation and NLP feature inspection

## Technologies
- Python 3.11
- AssemblyAI API
- Hugging Face Transformers
- PyTorch (GPU-enabled)
- spaCy
- scikit-learn
- TQDM
- Matplotlib

## How to Set Up the Environment

### Option 1: Using Conda (Recommended)

Create the environment from the included environment file:

```bash
conda env create -f environment.yml
conda activate Pipeline
```

### Option 2: Using pip

Create a virtual environment and install dependencies from requirements.txt:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Unix/macOS

pip install -r requirements.txt
```

### Download Language Data

```bash
python -m nltk.downloader stopwords vader_lexicon
python -m spacy download en_core_web_sm
```

## How to Run the Pipeline

1. Open `Pipeline.ipynb` in Jupyter or VSCode
2. Set your AssemblyAI API key:
```python
API_KEY = "your_api_key_here"
```
3. Add your `.mp3` files
4. Run the notebook cells step by step:
   - Upload and transcribe
   - Translate
   - Extract NLP features
   - Classify emotions
   - Save the final CSV

## Audio File Fixing (if needed)

Use this command to re-encode broken mp3 files:

```bash
ffmpeg -i "input.mp3" -acodec libmp3lame -ab 192k -ar 44100 "fixed_output.mp3"
```

## Notes
- Tested on Python 3.11 with GPU (CUDA-enabled)
- Timestamps align best with AssemblyAI using `format_text=True`
