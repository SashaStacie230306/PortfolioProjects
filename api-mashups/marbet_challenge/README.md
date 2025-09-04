# Marbet Trip Assistant — RAG Chatbot (Portfolio Cut)

## What I built

I built a **travel concierge chatbot** for Marbet’s incentive trip: a Streamlit app that uses **RAG** (Chroma vector DB + Ollama embeddings) with **Llama 3** for answers grounded in curated trip documents (itinerary, spa, Wi-Fi, packing, A–Z ship guide, visa info). It includes a small **test harness** and a report with results and next steps.

---

## Highlights

* **Streamlit chat UI** with light/dark theme, session chat history, and JSONL logging. A small **visa fallback** ensures a safe, useful answer even if retrieval misses.&#x20;
* **RAG stack:** `OllamaEmbeddings("mxbai-embed-large")` + **Chroma** persistent store (`chroma_db`) + **Llama 3** (`llama3` / `llama3.2:3b`).
* **Vector build script** that parses/normalizes JSON sources (e.g., ESTA/eTA, activities, spa services, Wi-Fi steps) and chunk-splits before embedding.&#x20;
* **CLI examples** for quick retrieval QA and a **keyword-based test runner** to score typical guest questions.
* **Project report** with performance notes (great on Wi-Fi/spa, weaker on vague/out-of-scope), and a recommendation to keep future content in **structured JSON/Markdown** instead of PDFs.&#x20;

---

## Repo contents

```
marbet_challenge/
├─ app.py                     # Streamlit chat app (RAG + UI + visa fallback + logging)
├─ vector.py                  # build/update Chroma vector store from curated JSON
├─ main.py                    # minimal RetrievalQA example (prints answer + sources)
├─ run_marbet_test.py         # small evaluation harness against test_cases.json
├─ check.ipynb                # scratch/verification notebook
├─ Marbet Trip Assistant.pdf  # project report (design, data prep, testing, results)
└─ challenge/                 # (place JSON knowledge files here; see below)
```

**Expected JSON files in `challenge/`:**
`esta_info.json`, `eta_info.json`, `extracted_activities_v2.json`, `packing_list.json`, `scenic_eclipse_cleaned.json`, `spa_services.json`, `wifi_setup_steps.json`. (The app/scripts currently read these names.)

---

## Quickstart

### 0) Prereqs

* Python 3.10+
* **Ollama** installed locally, with models:

  * `ollama pull llama3` (and/or `llama3.2:3b`)
  * `ollama pull mxbai-embed-large`

```bash
pip install streamlit langchain langchain-ollama langchain-community langchain-chroma chromadb pymupdf
```

### 1) Put knowledge files in place

Create `marbet_challenge/challenge/` and drop the JSON files listed above.

> Note: `app.py` and scripts ship with **absolute Windows paths** for the logo and `challenge/`. Switch them to **relative paths** for your machine (search for `C:/Users/...` and adjust).&#x20;

### 2) Build the vector store

```bash
python vector.py
# creates/updates ./chroma_db with embedded chunks
```

(Uses a parser for ESTA/eTA and generic JSON → chunks → Chroma.)&#x20;

### 3) Run the app

```bash
streamlit run app.py
```

Try: “What should I pack?”, “Is Wi-Fi available on board?”, “What’s the schedule on Monday?” (Suggested prompts are in the UI.)&#x20;

### 4) (Optional) CLI check

```bash
python main.py
```

Prints an answer and source doc names for a sample query.&#x20;

### 5) (Optional) Accuracy smoke test

Create `test_cases.json` with items like:

```json
[
  {"id": 1, "category": "wifi", "prompt": "How do I connect to Wi-Fi?", "expected_keywords": ["WiFi", "steps", "Android", "iOS"]},
  {"id": 2, "category": "spa", "prompt": "What spa services are available?", "expected_keywords": ["massage", "facial", "prices"]}
]
```

Then run:

```bash
python run_marbet_test.py
```

Scores % of expected keywords matched.&#x20;

---

## How it works 

* **UI**: Streamlit chat with theme toggle, session history, and JSONL logging. A small `custom_logic()` injects a safe **visa** response if the query mentions visas.&#x20;
* **RAG**: JSON files → `Document` objects → **OllamaEmbeddings** → **Chroma** (persisted). Inference uses `ConversationalRetrievalChain` (chat history aware) or `RetrievalQA` (simple one-shot).
* **Prompting**: friendly, concise persona; if not confident, defer users to Marbet contact.&#x20;

---

## Results

* **Strong** on scoped, factual questions (Wi-Fi setup, spa services) — near 100% keyword match.
* **Weaker** on vague prompts (“What do I need?”, “What about Tuesday?”).
* Visa **fallback** works reliably; **out-of-scope** detection needs strengthening.
* Recommendation: maintain **structured JSON/CSV/Markdown** instead of PDFs to cut parsing overhead and improve reliability.&#x20;

---

## Roadmap

* Add intent classification to catch **vague/out-of-scope** early.
* Enrich itinerary with **date-aware** reasoning (e.g., “Tuesday” → activities).
* Inline images/cards for steps and spa menus in the UI.
* Feedback loop from real chats to refine prompts and retrieval.&#x20;

---

## Privacy & attribution

The knowledge base is compiled from Marbet’s trip materials; I do not redistribute proprietary PDFs. The app reads only your local JSON copies.
