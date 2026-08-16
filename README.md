# RepoTriage

An AI-powered triage assistant for GitHub issues — retrieval-augmented Q&A and
triage reasoning over a repo's issue history.

## Architecture

```
GitHub issues ──ingest.py──► data/issues.parquet
                                    │
                              embed.py (sentence-transformers, local, free)
                                    ▼
                          Qdrant (vector search)
                                    │
                    qa.py / app.py │ retrieval + gpt-4o-mini reasoning
                                    ▼
                     Answer (qa) or priority/duplicate flag (triage)
                                    │
                          guardrails.py (input/output validation, rate limiting)
                                    │
                          data/query_log.csv (cost/latency observability)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in GITHUB_TOKEN, QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY
export $(cat .env | xargs)   # or use a tool like direnv / python-dotenv
```

## Day 1 — Ingest & clean ✅

```bash
python3 ingest.py --repo microsoft/vscode --max-issues 800
```
Outputs `data/issues.parquet` and `data/issues.csv`. **Already run** — the data folder
in this repo has 800 real issues from `microsoft/vscode` ready to use.

## Day 2 — Embeddings + Qdrant vector search

```bash
python3 embed.py --input data/issues.parquet --collection vscode_issues
```
Embeds every issue locally (`all-MiniLM-L6-v2`, no API cost) and uploads to a Qdrant
collection. Get a free Qdrant Cloud cluster at https://cloud.qdrant.io (1GB free tier
is plenty for this dataset), or run Qdrant locally via `docker run -p 6333:6333 qdrant/qdrant`.

## Day 3 — LLM Q&A + triage reasoning

```bash
python3 qa.py qa --question "What are common causes of crash reports?"
python3 qa.py triage --title "App crashes on save" \
  --body "When I save a large file the app freezes and crashes."
```
Every call logs token usage, cost, and latency to `data/query_log.csv`.

## Day 4 — FastAPI endpoints + guardrails + eval set

```bash
uvicorn app:app --reload --port 8000
```
Then visit http://localhost:8000/docs for interactive Swagger UI, or:
```bash
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are common causes of crash reports?"}'
```

**Guardrails** (`guardrails.py`): input length/emptiness checks, a basic prompt-injection
filter, per-IP rate limiting (20 req/min), and output schema validation on triage responses
so a malformed LLM response never reaches the client as a 200.

**Eval set** (`eval_set.json` + `eval.py`): a small hand-written set of QA and triage cases
with explicit expected properties (keyword presence, expected priority band, duplicate
detection) — deterministic checks rather than LLM-judges-LLM scoring. Run it with:
```bash
python3 eval.py
```

Guardrails have their own fast unit tests that need no API keys — see `tests/test_guardrails.py`
and `.github/workflows/ci.yml`.

## Day 5 — Deploy + document

```bash
docker build -t repotriage .
docker run -p 8000:8000 --env-file .env repotriage
```
Push the image to Render, Railway, or Fly.io (all have free/cheap tiers that support Docker
deploys) and point `QDRANT_URL` at your Qdrant Cloud cluster so the deployed API doesn't
depend on your local machine.

## Known limitations (intentionally left visible, not hidden)

- Stored data has no PII scrubbing beyond what GitHub already redacts — fine for public
  repos, would need review before pointing at a private repo.
- The rate limiter is in-memory and per-process — resets on restart and doesn't share
  state across multiple instances. Swap for Redis if this needs to scale horizontally.
- The prompt-injection filter is a basic keyword heuristic, not a real security boundary.
- `eval.py` uses deterministic keyword/schema checks rather than semantic scoring — good
  enough to catch regressions, not a substitute for human review of answer quality.
