# PortfolioProjects — Sasha Stacie

This folder is a tour of my work across **ML**, **GenAI/RAG**, and **Analytics/BI**. Each subfolder has its own README with what it is, what’s inside, and how to run it.

---

## What’s here

### `arboreal-cnn-xai/` — tree/plant recognition **with explanations**

A CNN for urban tree/plant classification with **Grad-CAM** heatmaps so every prediction comes with a visual “why.”

* **Why it exists:** support inventories and conservation with transparent decisions.
* **What’s inside:** training notebook, `src/xai.py` (Grad-CAM).
* **Skills:** PyTorch, CNNs, data augmentation, XAI, evaluation/visualization.

---

### `audio-to-emotion-nlp/` — speech → transcript → **per-sentence emotions**

Production-style pipeline: **AssemblyAI** for ASR, sentence segmentation with timestamps, and a fine-tuned **Hugging Face** model for emotion labels; optional **FastAPI** service.

* **Why it exists:** jump straight to the moments that matter in calls/interviews.
* **What’s inside:** CLI pattern, `checkpoint-*/`, `label_encoder.pkl`, CSV schema (`Start Time | End Time | Sentence | Emotion`), FastAPI app.
* **Skills:** Transformers, PyTorch, ASR integration, pipeline design, testing, FastAPI.

---

### `data-analysis-nac/` — NAC Breda player evaluation

Cleaned and analyzed scouting data with **goal-contribution signals** (shots/90, SOT%, assists/90, duels), sliced by position and age; concise final report.

* **Why it exists:** quick, data-backed shortlists and role-based flags.
* **What’s inside:** preprocessing, missing-value fixes, analysis notebook, final PDF.
* **Skills:** EDA, feature engineering, sports analytics, clear reporting.

---

### `powerbi-sdg-dashboard/` — SDG 6: Clean Water & Sanitation

One-screen **Power BI** report connecting access to safe water/WASH practices with mortality patterns.

* **Views:** global (access map + KPI), macro (**GDP vs deaths vs improved water** with a correlation toggle), micro (household WASH: source, storage, treatment, sanitation, handwashing).
* **Why it exists:** frame a public-health conversation—where the gaps are and why.
* **What’s inside:** `.pbix` file + exported screenshots.
* **Skills:** Power BI, data modeling, DAX, bookmarks/slicers, storytelling.

---

### `sme-genai-willingness-study/` — who’s ready to learn GenAI, and what helps?

Mixed-methods study (survey + short interviews). I led the quant: descriptives, correlations, **t-tests/ANOVA**, simple regression; co-wrote policy recommendations.

* **Key signal:** **role relevance** and **manager support** matter most; degree level was not significant in my sample. Barriers: time, unclear use cases, privacy/compliance.
* **What’s inside:** `analysis.ipynb`, final research paper, individual sub-study, policy paper, project roadmap.
* **Skills:** statistical testing, research methods, visualization, policy writing.

---

### `time-series-forecasting/` — clean baselines & readable metrics

Straightforward **ARIMA** baselines with **MAPE/SMAPE** helpers and a clear forecast chart; short write-up.

* **Why it exists:** a dependable baseline before adding seasonal/ML hybrids.
* **Skills:** time-series modeling, error metrics, notebook hygiene.

---

### `api-mashups/` — small builds where I wire tools together

* **`calculator/`** — one-file **Streamlit** app with a pure Python core (add/subtract/multiply/divide).
  *Skills:* Streamlit, tiny app structure.
* **`football/`** — **player value score** from normalized indicators + **home W/D/L** classifier (RF/XGBoost) with engineered match features; tidy CSV outputs.
  *Skills:* feature engineering, model comparison, reproducible exports.
* **`gen_ai_challenge/`** — storyboard generation with **ComfyUI** and **Automatic1111/WebUI**, prompt iteration via **Ollama**, **character consistency** (fixed seed + anchor phrase).
  *Skills:* generative imaging, prompt design, workflow tooling.
* **`marbet_challenge/`** — **Streamlit** travel assistant using RAG (**Chroma + Ollama embeddings**, **Llama 3**) over curated JSON (itinerary, Wi-Fi, spa, packing, visa). Includes a visa fallback and a tiny test harness.
  *Skills:* RAG, embeddings, lightweight evaluation, UX basics.

---

### `completed-courses/` — certificates & topic snapshots

PDFs from short courses I completed alongside my degree.

* **Highlights:** Power BI (modeling, prep, viz, EDA, churn, report design), Python/ML (pandas, supervised learning, tree-based, clustering, dim. reduction, time-series), Deep Learning (Keras CNN), GenAI/Systems (OpenAI systems, AI in business), foundations (UX/Interaction/Design Thinking, SQL).
* **Use:** quick proof of topics refreshed; a reference index when I revisit a concept.

---

## How to run (most Python projects)

```bash
# 1) Create & activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# 2) Upgrade pip and install dependencies (see each project README)
pip install -U pip
# example:
# pip install -r requirements.txt

# 3) Open notebooks or run the app
jupyter lab        # for notebooks
# or
uvicorn app:app --reload   # FastAPI (check module path in the project README)
streamlit run app.py       # Streamlit apps
```

> **Data & keys:** I don’t commit private datasets or secrets. Projects that need them include an `.env.example` and instructions in their README.

---

## Repo structure (top level)

```
PortfolioProjects/
├─ arboreal-cnn-xai/
├─ audio-to-emotion-nlp/
├─ data-analysis-nac/
├─ powerbi-sdg-dashboard/
├─ sme-genai-willingness-study/
├─ time-series-forecasting/
├─ api-mashups/
│  ├─ calculator/
│  ├─ football/
│  ├─ gen_ai_challenge/
│  └─ marbet_challenge/
└─ completed-courses/
```

---

## Tools I use a lot

**Python** (pandas, numpy, scikit-learn), **PyTorch**, **Hugging Face**, **FastAPI**, **Streamlit**, **Power BI** (DAX, modeling), and for GenAI/RAG: **Ollama**, **Chroma**, **ComfyUI**, **Automatic1111/WebUI**.

---