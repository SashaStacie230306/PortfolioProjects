# Emotion Classification CLI

This CLI allows users to classify emotions from **text**, **audio**, or **video** inputs using a modular NLP pipeline. It's designed for ease of use and can run in **mock mode** for testing without making real API calls.

## 🚀 Features

- 📄 Classify emotions from raw **text**
- 🔊 Extract and classify emotions from **audio** files (`.wav`, `.mp3`)
- 🎥 Extract and classify emotions from **video** files (`.mp4`, `.mov`)
- 🌐 Process media from a **URL** (e.g., YouTube, MP3)
- 🧪 **Mock mode** support for dry runs
- 📦 Saves results to CSV format in an `output/` directory

---

## 🛠️ Installation

1. **Clone this repository:**
   ```bash
   git clone <repo-url>
   cd <repo-dir>
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Set up API key for AssemblyAI:**
   ```bash
   export ASSEMBLYAI_API_KEY="your-api-key-here"
   ```

---

## 🧪 Mock Mode (Testing)

To run in **mock mode** without real predictions:
```bash
export SPHINX_MOCK_MODE=1
```

---

## 📦 Usage

Run via the command line:
```bash
python cli.py [COMMAND] [ARGS]
```

### 🔤 Classify Text
```bash
python cli.py text "I'm so happy to see you!"
```
📁 Output: `output/text_prediction.csv`

---

### 🔈 Classify Audio
```bash
python cli.py audio path/to/audio.wav
```
📁 Output: `output/final_predictions.csv`

---

### 🎞️ Classify Video
```bash
python cli.py video path/to/video.mp4
```
📁 Output: `output/final_predictions.csv`

---

### 🌐 Classify from URL
```bash
python cli.py url "https://link-to-media-file.com/audio.mp3"
```
📁 Output: `output/final_predictions.csv`

Optional `--language` argument:
```bash
python cli.py url "https://..." --language "en"
```

---

## 📝 Output Format

**Text Output (CSV):**

| Input Text              | Predicted Emotion | Confidence |
|------------------------|-------------------|------------|
| I'm excited for today! | happy             | 0.94       |

**Media Output (CSV):**

| Start Time | Translation | Emotion | Confidence (%) |
|------------|-------------|---------|----------------|
| 00:00      | Hello       | neutral | 0.0            |

---

## 🧱 Project Structure

```
cli.py                    # Main CLI entry point
app/predict.py            # Core prediction logic
utils/logger.py           # Logging utilities
output/                   # Output CSVs
```

---

## ⚠️ Requirements

- Python 3.8+
- `ffmpeg` installed and available in PATH (for video/audio extraction)
- AssemblyAI API key (for real audio/video processing)

---
