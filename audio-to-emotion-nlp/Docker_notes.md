# Emotion Classification API Local Deployment Guide

A FastAPI REST API for predicting emotion from text, uploading audio/video files for transcription and emotion classification, processing YouTube/media links, and serving downloadable CSV predictions.

---

## Project Structure

```bash
project/
├── docker/
│   └── Dockerfile          # API Dockerfile
├── docker-compose.yml      # Multi-service deployment
├── AUI_Dockerfile          # Frontend (Gradio UI) Dockerfile
├── pyproject.toml          # Poetry dependencies
├── poetry.lock             # Poetry lockfile
├── .env                    # Environment variables & secrets
├── src/
│   ├── app/
│   │   ├── main.py         # FastAPI app entrypoint
│   │   ├── predict.py      # Emotion classification logic
│   │   └── ...             # Other modules
│   └── demo/
│       ├── app_ui.py       # Gradio UI
│       └── requirements.txt# UI dependencies
└── temp_uploads/           # File uploads mount
```

---

## Quick Start

### 1. Clone Repo

```bash
git clone https://github.com/BredaUniversityADSAI/2024-25d-fai2-adsai-group-nlp7
cd nlp7
```

### 2. Configure Environment

Create a `.env` in project root with your keys:

```
API_KEY=...
LW_CLIENT_ID=...
```

### 3. Build and Run (Docker Compose)

```bash
docker-compose up --build
```

* **Backend:** [http://localhost:3126/docs](http://localhost:3126/docs) (FastAPI Swagger)
* **Frontend:** [http://localhost:3127](http://localhost:3127) (Gradio UI)

### 4. Manual Docker

#### Build Images

```bash
docker build -f docker/Dockerfile -t nlp-api:latest .
docker build -f AUI_Dockerfile -t nlp-ui:latest .
```

#### Run API

```bash
docker run -p 8000:8000 --env-file .env -v "$(pwd)/temp_uploads:/app/temp_uploads" nlp-api:latest
```

#### Run UI

```bash
docker run -p 7860:7860 -e API_URL=http://host.docker.internal:8000 nlp-ui:latest
```

---

## Docker Compose

```yaml
version: '3.8'
services:
  backend:
    image: celinewu1/nlp-api:1.3
    platform: linux/amd64
    ports:
      - "3126:8000"
    restart: unless-stopped
    environment:
      - PYTHONPATH=/app/src
    volumes:
      - ./temp_uploads:/app/temp_uploads

  frontend:
    build:
      context: .
      dockerfile: AUI_Dockerfile
    ports:
      - "3127:7860"
    environment:
      - API_URL=http://backend:8000
    depends_on:
      - backend
```

---

## Dockerfiles

### API (`docker/Dockerfile`)

```dockerfile
FROM python:3.10
WORKDIR /app

ENV CACHE_BUSTER=20250612
LABEL maintainer="231265@buas.nl"

RUN pip install --upgrade pip \
    && pip install poetry

COPY pyproject.toml poetry.lock* /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-root

RUN pip install torch==2.7.0+cpu --index-url https://download.pytorch.org/whl/cpu

COPY src/checkpoint-3906 /app/src/checkpoint-3906
COPY src /app/src
COPY .env /app/.env

ENV PYTHONPATH="/app/src"
EXPOSE 8000
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### UI (`AUI_Dockerfile`)

```dockerfile
FROM python:3.10
WORKDIR /app

COPY src/demo/requirements.txt ./
RUN pip install --upgrade pip \
    && pip install gradio>=4.65.0 gradio-client>=0.4.4 \
    && pip install -r requirements.txt

COPY src/demo/ .
CMD ["python", "app_ui.py"]
```

---

## API Endpoints

* **GET /** returns `{"message":"Hello World"}`
* **GET /health** returns `{"status":"healthy"}`
* **POST /predict\_text** body `{ "text": "..." }` → `{ "emotion": "joy" }`
* **POST /upload\_audio** (form file) → transcription + emotion
* **POST /predict\_youtube** body `{ "url": "..." }` → transcription + emotion
* **GET /download\_text\_csv** downloads text results CSV
* **GET /download\_csv** downloads media results CSV

---

## Troubleshooting

### Container Exits with Code 1

Check container logs for Poetry or `.env` errors:

```bash
docker logs <container-name>
```

### Healthcheck Fails

Ensure the health endpoint is defined in `main.py`:

```python
@app.get("/health")
async def health():
    return {"status": "healthy"}
```

---

## Deployment on Portainer

Use Portainer to deploy via your Git repository:

1. **Push to GitHub** (private or public repo)
2. In Portainer, go to **Stacks → Add Stack**
3. Select **Git Repository** and enter:

   * **Repository URL:** `https://github.com/your-username/emotion-api`
   * **Branch:** `main`
   * **Compose path:** `docker-compose.yml`
   * **Authentication:** Enable (use your Git username & token, e.g. *deuza*)
   * **Skip TLS verification:** Enable if needed
4. Click **Deploy the Stack**
5. Access your app at: `http://<your-server-ip>:<your-port>/docs`

> **Credentials for Portainer Access**
>
> | Field    | Value                                          |
> | -------- | ---------------------------------------------- |
> | Group    | NLP7                                           |
> | Server   | Lambda\_0                                      |
> | IP       | 194.171.191.226                                |
> | Port     | 3126–3129                                      |
> | GPU      | 5                                              |
> | Volume   | `/data/docker_volumes/portainer/2025_y2d/NLP7` |
> | Username | `nlp7`                                         |
> | Password | Ask Myrthe                                     |
