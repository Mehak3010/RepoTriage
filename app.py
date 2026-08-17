"""
RepoTriage -- FastAPI endpoints + guardrails + multi-repo support

Wraps qa.py's retrieval + LLM logic in a small HTTP API, with input/output
guardrails and per-IP rate limiting applied at the edge. Can serve any number
of repos at once -- add a new one via POST /repos, then target it in /qa or
/triage with the "collection" field returned from that call.

Setup:
    pip install fastapi "uvicorn[standard]"
    export GROQ_API_KEY="your_groq_key_here"
    export GITHUB_TOKEN="your_github_token_here"   # needed to add new repos
    export QDRANT_URL="https://xxxx.cloud.qdrant.io"
    export QDRANT_API_KEY="your_qdrant_key_here"

Run:
    uvicorn app:app --reload --port 8000

Add a new repo (runs in the background -- poll the returned job_id):
    curl -X POST http://localhost:8000/repos \
      -H "Content-Type: application/json" \
      -d '{"repo": "facebook/react", "max_issues": 500}'

    curl http://localhost:8000/repos/jobs/<job_id>

List repos already ingested:
    curl http://localhost:8000/repos

Ask/triage against a specific repo:
    curl -X POST http://localhost:8000/qa \
      -H "Content-Type: application/json" \
      -d '{"question": "...", "collection": "facebook_react"}'

Docs: once running, visit http://localhost:8000/docs for interactive Swagger UI.
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from guardrails import GuardrailError, RateLimiter, validate_text_input, validate_triage_output
from qa import answer_question, get_clients, triage_issue
from repo_manager import add_repo, list_repos, repo_to_collection_name

DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "vscode_issues")
TOP_K = int(os.environ.get("TOP_K", "5"))

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

# Loaded once at startup so each request doesn't pay the model-load / connection cost.
clients: dict = {}

# In-memory job tracker for background repo-ingest jobs. Fine for a single-instance
# portfolio deployment -- resets on restart, doesn't share state across instances.
jobs: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # get_clients() also validates the given collection exists -- pass the default
        # but note it's no longer the *only* collection this API can serve.
        qdrant, openai_client, embed_model = get_clients(DEFAULT_COLLECTION)
        clients["qdrant"] = qdrant
        clients["openai"] = openai_client
        clients["embed_model"] = embed_model
        print(f"Startup OK -- default collection '{DEFAULT_COLLECTION}'")
    except SystemExit:
        raise RuntimeError(
            "Failed to initialize clients. Check QDRANT_URL, GROQ_API_KEY, and that "
            f"the '{DEFAULT_COLLECTION}' collection exists (run embed.py first, or "
            "add a repo via POST /repos once the server is up)."
        )
    yield
    clients.clear()


app = FastAPI(
    title="RepoTriage API",
    description="Retrieval-augmented Q&A and triage over any public GitHub repo's issues.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed frontend URL once you have it
    allow_methods=["*"],
    allow_headers=["*"],
)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    collection: Optional[str] = Field(default=None, description="Which ingested repo to query. Defaults to the server's default collection.")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class QAResponse(BaseModel):
    question: str
    answer: str
    collection: str


class TriageRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    body: str = Field(..., min_length=1, max_length=5000)
    collection: Optional[str] = Field(default=None)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class TriageResponse(BaseModel):
    priority: str
    is_duplicate: bool
    duplicate_of: Optional[int]
    reasoning: str
    collection: str


class AddRepoRequest(BaseModel):
    repo: str = Field(..., min_length=3, max_length=200, description="owner/name, e.g. facebook/react")
    max_issues: int = Field(default=500, ge=10, le=2000)


class AddRepoResponse(BaseModel):
    job_id: str
    status: str


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _resolve_collection(requested: Optional[str]) -> str:
    """Validate the requested collection exists in Qdrant, or fall back to default."""
    collection = requested or DEFAULT_COLLECTION
    existing = [c.name for c in clients["qdrant"].get_collections().collections]
    if collection not in existing:
        raise GuardrailError(
            f"Collection '{collection}' not found. Available: {existing or '(none yet)'}. "
            "Add a repo first via POST /repos."
        )
    return collection


@app.exception_handler(GuardrailError)
async def guardrail_exception_handler(request: Request, exc: GuardrailError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "default_collection": DEFAULT_COLLECTION}


@app.get("/repos")
def repos_endpoint():
    """List every repo currently ingested and ready to query."""
    return {"repos": list_repos(clients["qdrant"])}


def _run_add_repo_job(job_id: str, repo: str, max_issues: int):
    def progress(msg: str):
        jobs[job_id]["message"] = msg

    try:
        jobs[job_id]["status"] = "running"
        result = add_repo(repo, max_issues, clients["embed_model"], clients["qdrant"], progress_cb=progress)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)


@app.post("/repos", response_model=AddRepoResponse)
def add_repo_endpoint(req: AddRepoRequest, request: Request, background_tasks: BackgroundTasks):
    rate_limiter.check(_client_ip(request))

    if "/" not in req.repo:
        raise GuardrailError("repo must be in 'owner/name' format, e.g. 'facebook/react'.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "message": "Queued", "repo": req.repo, "result": None}
    background_tasks.add_task(_run_add_repo_job, job_id, req.repo, req.max_issues)

    return AddRepoResponse(job_id=job_id, status="queued")


@app.get("/repos/jobs/{job_id}")
def repo_job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest, request: Request):
    rate_limiter.check(_client_ip(request))
    validate_text_input(req.question, "question")
    collection = _resolve_collection(req.collection)

    answer = answer_question(
        clients["qdrant"], clients["openai"], clients["embed_model"],
        collection, req.question, req.top_k or TOP_K,
    )
    return QAResponse(question=req.question, answer=answer, collection=collection)


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(req: TriageRequest, request: Request):
    rate_limiter.check(_client_ip(request))
    validate_text_input(req.title, "title")
    validate_text_input(req.body, "body")
    collection = _resolve_collection(req.collection)

    result = triage_issue(
        clients["qdrant"], clients["openai"], clients["embed_model"],
        collection, req.title, req.body, req.top_k or TOP_K,
    )

    if "error" in result:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON. Try again.")

    validate_triage_output(result)
    return TriageResponse(**result, collection=collection)