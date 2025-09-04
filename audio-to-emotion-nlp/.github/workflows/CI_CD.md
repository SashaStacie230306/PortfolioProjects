@ -0,0 +1,225 @@
# Project Automation Guide — GitHub Actions, Pre-commit & Docker CI/CD

## Purpose

This project uses **GitHub Actions**, **pre-commit hooks**, and **Docker** to automatically check code quality, run tests, build images, and deploy the application every time someone pushes code or opens a pull request (PR).

The goal is to **prevent broken, unformatted, or inconsistent code** from being merged into the main branch and to **automate the entire deployment pipeline** — helping us all work more efficiently.

## What Happens Automatically

### 1. Pre-commit Checks (Linting & Formatting)
We use tools like:

- **black** – format code
- **flake8** – style linting
- **isort** – organize imports
- **autoflake** – remove unused imports
- **docformatter** – format docstrings
- **check-yaml**, **end-of-file-fixer**, etc.

These run automatically:
- Locally (when you use `pre-commit`)
- On GitHub for changed files in each PR

### 2. Automated Testing

GitHub runs a workflow that:
- Installs dependencies via **Poetry**
- Uses Python 3.10
- Runs the test suite using:
  ```bash
  poetry run pytest -n auto --maxfail=1 --disable-warnings -v
  ```

### 3. Docker Image Building & Publishing

Our CI/CD pipeline automatically:
- **Builds Docker images** for both backend (API) and frontend (UI)
- **Tags images** with git SHA for traceability
- **Pushes to Docker Hub** (`celinewu1/nlp-api` and `celinewu1/nlp-ui`)
- **Creates `:latest` tags** only when code is merged to main
- **Builds for linux/amd64** platform for consistency

## What You Should Do as a Developer

### 1. Initial Setup

```bash
poetry install
pre-commit install
```

### 2. Before Committing Code
**If you're using the terminal: FYI Sometimes you need to run it 2 times so the issues are corrected**
```bash
poetry run pre-commit run --all-files
```
Fix any issues that are printed, then stage and commit again.

**If you're using GitHub Desktop:**
1. Open your terminal in the project folder.
2. Run:
   ```bash
   poetry run pre-commit run --all-files
   ```
3. Then go back to GitHub Desktop to:
   - Commit and push your code
   - Open a Pull Request

### 3. Pushing Changes & Creating a PR
```bash
git add .
git commit -m "your message"
git push origin your-branch-name
```
Then go to GitHub and create a Pull Request (PR) from your branch.

## What Happens on GitHub (CI/CD)

### On Every Branch Push:
GitHub will:

1. **Run pre-commit on only changed files:**
   ```bash
   poetry run pre-commit run --from-ref origin/main --to-ref HEAD
   ```

2. **Run the test suite:**
   ```bash
   poetry run pytest -n auto --maxfail=1 --disable-warnings -v
   ```

3. **Build and push Docker images** tagged with git SHA:
   - `celinewu1/nlp-api:abc123`
   - `celinewu1/nlp-ui:abc123`

### On Main Branch (after PR merge):
Additionally:
4. **Tag and push as `:latest`:**
   - `celinewu1/nlp-api:latest`
   - `celinewu1/nlp-ui:latest`

## Testing the Application

### Local Development Testing
```bash
# Start the application locally
docker-compose up -d

# Test backend API
curl http://localhost:3126
# Should return: {"message":"Welcome to the Text, Audio & Video Emotion Classifier API!"}

# Test frontend
open http://localhost:3127
# Opens the Gradio UI in your browser

# Check container status
docker-compose ps

# View logs if needed
docker-compose logs -f
```

### Testing Docker Hub Images
```bash
# Test with fresh images from Docker Hub
docker-compose down
docker system prune -f

# Pull latest images
docker pull celinewu1/nlp-api:latest
docker pull celinewu1/nlp-ui:latest

# Start again
docker-compose up -d
```

### API Endpoints
- **Backend API**: `http://localhost:3126`
- **Frontend UI**: `http://localhost:3127`
- **API Docs**: `http://localhost:3126/docs`

## Branch Protection Rules
To protect main, we have enforced:

- All pre-commit checks must pass
- All tests must pass
- All Docker builds must succeed
- Code review approval

This ensures our main branch is always clean, functional, and deployable.

## Docker Architecture

### Services
- **Backend**: FastAPI-based emotion classifier API
- **Frontend**: Gradio-based web interface
- **Platform**: Built for `linux/amd64` for deployment consistency

### Images
- `celinewu1/nlp-api:latest` - Backend API service
- `celinewu1/nlp-ui:latest` - Frontend UI service

## Tools Used

| Tool | Purpose |
|------|---------|
| Poetry | Dependency management |
| pre-commit | Linting, formatting, cleanup |
| pytest | Unit/integration test execution |
| pytest-xdist | Run tests in parallel (faster) |
| Docker & Docker Compose | Containerization and orchestration |
| GitHub Actions | CI/CD automation |
| Docker Hub | Image registry |

## Example: Common Local Commands

```bash
# Install everything (first time)
poetry install
pre-commit install

# Run code cleanup checks on all files
poetry run pre-commit run --all-files

# Run tests manually
poetry run pytest

# Run tests in parallel (same as GitHub does)
poetry run pytest -n auto --maxfail=1 --disable-warnings -v

# Build Docker images locally
docker buildx build --platform linux/amd64 -f API_Dockerfile -t celinewu1/nlp-api:latest .
docker buildx build --platform linux/amd64 -f AUI_Dockerfile -t celinewu1/nlp-ui:latest .

# Start application
docker-compose up -d

# Stop application
docker-compose down

# Add + commit + push to remote branch
git add .
git commit -m "Your message here"
git push origin your-feature-branch
```

## Troubleshooting

### Platform Issues
If you get platform architecture errors:
```bash
# Rebuild for linux/amd64
docker buildx build --platform linux/amd64 -f API_Dockerfile -t celinewu1/nlp-api:latest .
docker buildx build --platform linux/amd64 -f AUI_Dockerfile -t celinewu1/nlp-ui:latest .
```

### Port Conflicts
If ports 3126 or 3127 are in use, update `docker-compose.yml`:
```yaml
ports:
  - "YOUR_PORT:8000"  # for backend
  - "YOUR_PORT:7860"  # for frontend
```
