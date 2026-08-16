"""
FastAPI endpoints + guardrails

Wraps qa.py's retrieval + LLM logic in a small HTTP API, with input/output
guardrails and per-IP rate limiting applied at the edge.

Setup:
    pip install fastapi "uvicorn[standard]"
    export GROQ_API_KEY="your_groq_key_here"
    export QDRANT_URL="https://xxxx.cloud.qdrant.io"
    export QDRANT_API_KEY="your_qdrant_key_here"

Run:
    uvicorn app:app --reload --port 8000

Try it:
    curl -X POST http://localhost:8000/qa \
      -H "Content-Type: application/json" \
      -d '{"question": "What are common causes of crash reports?"}'

    curl -X POST http://localhost:8000/triage \
      -H "Content-Type: application/json" \
      -d '{"title": "App crashes on save", "body": "Saving a large file freezes and crashes the app."}'

Docs: once running, visit http://localhost:8000/docs for interactive Swagger UI.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from guardrails import GuardrailError, RateLimiter, validate_text_input, validate_triage_output
from qa import answer_question, get_clients, triage_issue

from dotenv import load_dotenv
load_dotenv()

COLLECTION = os.environ.get("QDRANT_COLLECTION", "vscode_issues")
TOP_K = int(os.environ.get("TOP_K", "5"))

rate_limiter = RateLimiter(max_requests=20, window_seconds=60)

# Loaded once at startup so each request doesn't pay the model-load / connection cost.
clients: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        qdrant, openai_client, embed_model = get_clients(COLLECTION)
        clients["qdrant"] = qdrant
        clients["openai"] = openai_client
        clients["embed_model"] = embed_model
        print(f"Startup OK — connected to Qdrant collection '{COLLECTION}'")
    except SystemExit:
        # get_clients() calls sys.exit() on missing config/collection — convert that
        # into a clear startup failure instead of silently leaving `clients` empty.
        raise RuntimeError(
            "Failed to initialize clients. Check QDRANT_URL, GROQ_API_KEY, and that "
            f"the '{COLLECTION}' collection exists (run embed.py first)."
        )
    yield
    clients.clear()


app = FastAPI(
    title="RepoTriage API",
    description="Retrieval-augmented Q&A and triage over a repo's GitHub issues.",
    version="0.1.0",
    lifespan=lifespan,
)


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class QAResponse(BaseModel):
    question: str
    answer: str


class TriageRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    body: str = Field(..., min_length=1, max_length=5000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class TriageResponse(BaseModel):
    priority: str
    is_duplicate: bool
    duplicate_of: Optional[int]
    reasoning: str


def _client_ip(request: Request) -> str:
    # Behind a real reverse proxy you'd trust X-Forwarded-For here; for a
    # portfolio deployment behind e.g. Render/Railway's own proxy this is fine.
    return request.client.host if request.client else "unknown"


@app.exception_handler(GuardrailError)
async def guardrail_exception_handler(request: Request, exc: GuardrailError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION}


@app.post("/qa", response_model=QAResponse)
def qa_endpoint(req: QARequest, request: Request):
    rate_limiter.check(_client_ip(request))
    validate_text_input(req.question, "question")

    answer = answer_question(
        clients["qdrant"], clients["openai"], clients["embed_model"],
        COLLECTION, req.question, req.top_k or TOP_K,
    )
    return QAResponse(question=req.question, answer=answer)


@app.post("/triage", response_model=TriageResponse)
def triage_endpoint(req: TriageRequest, request: Request):
    rate_limiter.check(_client_ip(request))
    validate_text_input(req.title, "title")
    validate_text_input(req.body, "body")

    result = triage_issue(
        clients["qdrant"], clients["openai"], clients["embed_model"],
        COLLECTION, req.title, req.body, req.top_k or TOP_K,
    )

    if "error" in result:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON. Try again.")

    validate_triage_output(result)
    return TriageResponse(**result)