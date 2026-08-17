# RepoTriage

RepoTriage is an AI assistant for GitHub issues. Point it at any public repo,
and it can:

- **Answer questions** about that repo's issue history in plain English
  ("what are common causes of crash reports?"), citing specific issue numbers
- **Triage a new issue** you're about to file: it suggests a priority
  (high/medium/low) and flags if it looks like a duplicate of something
  already reported

It works on any public GitHub repo, not just one fixed repo. Add a repo
through the web UI (or the API) and it's ready to query within a couple of
minutes.

## How it works (short version)

1. Issues get pulled from GitHub's API and cleaned up
2. Each issue is turned into a vector embedding (a numeric fingerprint of its
   meaning) and stored in Qdrant, a vector database, so similar issues can be
   found by meaning, not just keyword matching
3. When you ask a question or submit an issue to triage, RepoTriage finds the
   most similar existing issues and hands them to an LLM (via Groq) along
   with your question, asking it to answer using only that context
4. You get an answer with issue numbers cited, or a priority/duplicate
   recommendation with reasoning

## What you need before you start

Three free accounts, no credit card required for any of them:

| What | Why | Where to get it |
|---|---|---|
| GitHub personal access token | To fetch issues from GitHub's API | https://github.com/settings/tokens -- classic token, `public_repo` scope is enough |
| Qdrant Cloud account | Free vector database to store issue embeddings | https://cloud.qdrant.io -- free tier (1GB) is plenty |
| Groq API key | Free, fast LLM inference for answering questions | https://console.groq.com/keys |

You'll also need Python 3.10+ and Node.js 18+ installed.

## Setup

### 1. Backend

```bash
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and fill in the four values from the table above:

```
GITHUB_TOKEN=your_github_token
QDRANT_URL=https://xxxx.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Then start the API:

```bash
uvicorn app:app --reload --port 8000
```

Visit http://localhost:8000/docs to confirm it's running -- you should see
interactive API documentation (Swagger UI).

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Visit http://localhost:5173 -- you should see "api online" in the top right
if the backend is reachable.

## Using it

### Add a repo

Click **+ add repo** at the top, type an `owner/name` (e.g. `facebook/react`),
and click **ingest**. This takes 1-3 minutes depending on repo size -- it's
fetching issues from GitHub and computing embeddings for each one. Progress
messages show while it runs. Once done, the repo appears in the dropdown and
is ready to query.

You can add as many repos as you like. Each one gets its own isolated data --
asking a question with `facebook/react` selected only searches that repo's
issues, not any other repo you've added.

### Ask a question

Pick a repo from the dropdown, go to the **Ask** tab, type a question, and
submit. You'll get an answer grounded in that repo's actual issues, with
issue numbers cited.

### Triage an issue

Go to the **Triage** tab, fill in a title and description for a
(hypothetical or real) new issue, and submit. You'll get back a suggested
priority, whether it looks like a duplicate of an existing issue, and the
model's reasoning.

## Using the API directly

Everything the frontend does is also available as a plain HTTP API --
useful if you want to script it or integrate it elsewhere.

```bash
# List repos already ingested
curl http://localhost:8000/repos

# Add a new repo (runs in the background)
curl -X POST http://localhost:8000/repos \
  -H "Content-Type: application/json" \
  -d '{"repo": "vitejs/vite", "max_issues": 500}'
# returns a job_id -- poll its status:
curl http://localhost:8000/repos/jobs/<job_id>

# Ask a question against a specific repo
curl -X POST http://localhost:8000/qa \
  -H "Content-Type: application/json" \
  -d '{"question": "What are common causes of crash reports?", "collection": "facebook_react"}'

# Triage an issue against a specific repo
curl -X POST http://localhost:8000/triage \
  -H "Content-Type: application/json" \
  -d '{"title": "App crashes on save", "body": "Saving a large file freezes and crashes the app.", "collection": "facebook_react"}'
```

If you omit `collection`, it falls back to the server's default repo (set
via `QDRANT_COLLECTION` in `.env`).

## Command-line tools (for development/testing)

These are the original scripts each piece was built with -- useful for
testing a single piece in isolation without running the full API.

```bash
# Ingest issues from a repo (the API's "+ add repo" does this automatically)
python3 ingest.py --repo microsoft/vscode --max-issues 800

# Embed and upload to Qdrant
python3 embed.py --input data/issues.parquet --collection vscode_issues

# Ask/triage from the command line
python3 qa.py qa --question "What are common causes of crash reports?"
python3 qa.py triage --title "App crashes on save" \
  --body "When I save a large file the app freezes and crashes."

# Run the eval set (checks answer quality against known-good expectations)
python3 eval.py

# Run the guardrails unit tests (no API keys needed)
python3 -m pytest tests/ -v
```

## Deployment

```bash
docker build -t repotriage .
docker run -p 8000:8000 --env-file .env repotriage
```

Push the image to Render, Railway, or Fly.io (all have free tiers that
support Docker) for the backend. For the frontend, deploy the `frontend/`
folder to Vercel or Netlify as a static site, setting `VITE_API_URL` to your
deployed backend's URL.

## Architecture

```
GitHub issues --ingest--> cleaned dataset
                                |
                          embed (sentence-transformers, local, free)
                                v
                    Qdrant (one collection per repo)
                                |
                qa.py / app.py | retrieval + Groq (openai/gpt-oss-120b) reasoning
                                v
                 Answer (ask) or priority/duplicate flag (triage)
                                |
                    guardrails.py (input/output validation, rate limiting)
                                |
                    data/query_log.csv (cost/latency observability)
```

## Troubleshooting

Common issues a first-time setup runs into, and how to fix them.

**`python3` is not recognized / opens the Microsoft Store (Windows)**

Windows sometimes maps `python3` to a Store shortcut instead of your actual
Python install. Use `python` instead, or `py` if that also fails:

```
python app.py
# or
py app.py
```

**`pip install` seems stuck / is taking a long time**

This is expected the first time -- `sentence-transformers` pulls in `torch`,
which is a large download (several hundred MB). Let it run; on a slow
connection this can take a few minutes. It's not actually frozen.

**`ERROR: missing environment variable(s): ...` even though your `.env` looks correct**

Two common causes:
- The `.env` file isn't actually named `.env` -- some editors (Notepad on
  Windows especially) silently save it as `.env.txt`. Check the real
  filename in a file explorer with extensions visible.
- You edited `.env` while the server was already running. Environment
  variables are only read once at startup -- stop the server (Ctrl+C) and
  restart it after any `.env` change.

**`model_not_found` / `does not exist or you do not have access to it`**

Groq periodically deprecates older models. If `GROQ_MODEL` in your `.env`
points at a model that's been retired, you'll get this error. Check
https://console.groq.com/docs/models for the current list of supported
models and update `GROQ_MODEL` accordingly, then restart the server.

**`401 Unauthorized` when adding a repo**

Your `GITHUB_TOKEN` is missing, expired, or lacks the right scope. Generate
a fresh classic token at https://github.com/settings/tokens with the
`public_repo` scope checked, update `.env`, and restart the server. You can
sanity-check a token directly:

```
curl -H "Authorization: Bearer YOUR_TOKEN" https://api.github.com/user
```

If that also returns 401, the token itself is invalid.

**Frontend shows "api offline" even though the backend terminal shows requests succeeding (200 OK)**

This is a CORS issue -- the browser is blocking the response even though the
server answered it. `app.py` already includes CORS middleware allowing all
origins, so this shouldn't happen on a fresh clone, but if you've modified
`app.py` and this comes back, make sure `CORSMiddleware` is still registered
on the `FastAPI` app instance.

**Frontend can't reach the backend at all**

Make sure `frontend/.env` (a *separate* file from the backend's `.env`) has
`VITE_API_URL=http://localhost:8000` and that you ran `cp .env.example .env`
inside the `frontend/` folder specifically, not just the project root.

**Copy-pasted code has broken syntax after pasting into your editor**

Some Windows editors mangle certain Unicode characters (em dashes, smart
quotes, arrows) on paste, silently truncating the rest of the line. If you
hit a syntax error right after pasting code, check the line the error
points to for anything that isn't a plain ASCII character.

## Known limitations (left visible on purpose, not hidden)

- Adding a repo blocks on GitHub's API rate limits like anything else using
  it -- a token gets you 5,000 requests/hour, which is plenty for normal use
  but can matter if you're adding many large repos back to back.
- No PII scrubbing beyond what GitHub already redacts -- fine for public
  repos.
- The rate limiter and background job tracker are in-memory and per-process
  -- they reset on restart and don't share state across multiple server
  instances. Fine for a single-instance deployment, would need Redis (or
  similar) to scale horizontally.
- The prompt-injection filter in `guardrails.py` is a basic keyword
  heuristic, not a real security boundary.
- `eval.py` uses deterministic keyword/schema checks rather than semantic
  quality scoring -- good for catching regressions, not a substitute for
  actually reading the answers.
