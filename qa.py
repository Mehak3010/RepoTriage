"""
RepoTriage — Day 3: LLM Q&A + lightweight triage reasoning

Two modes:
  1. qa      — ask natural-language questions, answered from retrieved issues
  2. triage  — given a new issue (title + body), find similar past issues and
               get a suggested priority / duplicate flag / reasoning

Every call logs its token usage and estimated cost to data/query_log.csv —
this is the seed of the "observability" story for your resume/interview.

Setup:
    pip install openai qdrant-client sentence-transformers
    export GROQ_API_KEY="your_groq_key_here"   # free at https://console.groq.com/keys
    export QDRANT_URL="https://xxxx.cloud.qdrant.io"
    export QDRANT_API_KEY="your_qdrant_key_here"

Uses Groq's OpenAI-compatible API (fast, generous free tier, open-weight models —
no OpenAI account needed). Override the model with GROQ_MODEL, e.g.:
    export GROQ_MODEL="llama-3.1-8b-instant"   # cheaper/faster, less capable

Usage:
    python qa.py qa --question "What are common causes of crash reports?"
    python qa.py triage --title "App crashes on save" --body "When I save a large file the app freezes and crashes."
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # must match embed.py — same vector space
LOG_PATH = "data/query_log.csv"

# Cache the embedding model across calls — SentenceTransformer load is slow (~1-2s), and
# app.py (Day 4) calls get_clients() per-request, so without this it'd reload every time.
_embed_model_cache: "SentenceTransformer | None" = None

# Groq pricing per 1M tokens (as of mid-2026) — update if Groq changes rates.
# Source: https://groq.com/pricing
GROQ_PRICING = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-4-scout-17b-16e-instruct": (0.11, 0.34),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}
COST_PER_1M_INPUT, COST_PER_1M_OUTPUT = GROQ_PRICING.get(MODEL_NAME, (0.59, 0.79))


def get_clients(collection: str):
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    missing = [name for name, val in [("QDRANT_URL", qdrant_url), ("GROQ_API_KEY", groq_key)] if not val]
    if missing:
        print(f"ERROR: missing environment variable(s): {', '.join(missing)}")
        sys.exit(1)

    global _embed_model_cache
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    # Groq's API is OpenAI-compatible, so we reuse the `openai` SDK and just point
    # base_url at Groq's endpoint. Variable is still called openai_client throughout
    # this file for minimal diff — it's really a Groq client.
    openai_client = OpenAI(api_key=groq_key, base_url=GROQ_BASE_URL)
    if _embed_model_cache is None:
        _embed_model_cache = SentenceTransformer(EMBED_MODEL_NAME)
    embed_model = _embed_model_cache

    if collection not in [c.name for c in qdrant.get_collections().collections]:
        print(f"ERROR: collection '{collection}' not found. Run embed.py first (Day 2).")
        sys.exit(1)

    return qdrant, openai_client, embed_model


def retrieve_similar(qdrant: QdrantClient, embed_model: SentenceTransformer, collection: str, text: str, top_k: int = 5) -> list[dict]:
    query_vector = embed_model.encode(text).tolist()
    results = qdrant.query_points(collection_name=collection, query=query_vector, limit=top_k).points
    return [{"score": r.score, **r.payload} for r in results]


def log_query(mode: str, prompt_tokens: int, completion_tokens: int, latency_s: float, question: str) -> None:
    os.makedirs("data", exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    cost = (prompt_tokens / 1_000_000 * COST_PER_1M_INPUT) + (completion_tokens / 1_000_000 * COST_PER_1M_OUTPUT)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "mode", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s", "query"])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            mode,
            prompt_tokens,
            completion_tokens,
            round(cost, 6),
            round(latency_s, 2),
            question[:200],
        ])
    print(f"  [logged: {prompt_tokens}+{completion_tokens} tokens, ~${cost:.5f}, {latency_s:.1f}s]")


def format_context(issues: list[dict]) -> str:
    blocks = []
    for issue in issues:
        blocks.append(
            f"Issue #{issue['issue_number']} [{issue['state']}, labels: {', '.join(issue['labels'])}]\n"
            f"Title: {issue['title']}\n"
            f"Body: {issue['body'][:500]}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(qdrant, openai_client, embed_model, collection: str, question: str, top_k: int = 5) -> str:
    similar = retrieve_similar(qdrant, embed_model, collection, question, top_k)
    context = format_context(similar)

    system_prompt = (
        "You are an assistant that answers questions about a software project's GitHub issues. "
        "Only use the issues provided in the context below. Always cite issue numbers (e.g. #1234) "
        "for any claim you make. If the context doesn't contain enough information to answer, say so "
        "clearly instead of guessing."
    )
    user_prompt = f"Context (retrieved issues):\n\n{context}\n\nQuestion: {question}"

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    latency = time.time() - t0

    answer = response.choices[0].message.content
    log_query("qa", response.usage.prompt_tokens, response.usage.completion_tokens, latency, question)

    print("\nRetrieved issues:")
    for issue in similar:
        print(f"  [{issue['score']:.3f}] #{issue['issue_number']}: {issue['title'][:70]}")

    return answer


def triage_issue(qdrant, openai_client, embed_model, collection: str, title: str, body: str, top_k: int = 5) -> dict:
    query_text = f"{title}\n\n{body}"
    similar = retrieve_similar(qdrant, embed_model, collection, query_text, top_k)
    context = format_context(similar)

    system_prompt = (
        "You are a triage assistant for a software project. Given a new issue and a list of "
        "similar existing issues, decide:\n"
        "1. priority: 'high', 'medium', or 'low'\n"
        "2. is_duplicate: true/false — is this very likely the same underlying problem as an existing issue?\n"
        "3. duplicate_of: the issue number if is_duplicate is true, otherwise null\n"
        "4. reasoning: 1-2 sentences explaining your decision, citing issue numbers where relevant\n\n"
        "Respond with ONLY a valid JSON object with these four keys. No markdown, no extra text."
    )
    user_prompt = (
        f"New issue:\nTitle: {title}\nBody: {body}\n\n"
        f"Similar existing issues:\n\n{context}"
    )

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    latency = time.time() - t0

    log_query("triage", response.usage.prompt_tokens, response.usage.completion_tokens, latency, title)

    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("WARNING: model did not return valid JSON. Raw output:")
        print(response.choices[0].message.content)
        result = {"error": "invalid_json", "raw": response.choices[0].message.content}

    print("\nSimilar issues considered:")
    for issue in similar:
        print(f"  [{issue['score']:.3f}] #{issue['issue_number']}: {issue['title'][:70]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="RepoTriage Day 3: Q&A and triage")
    parser.add_argument("--collection", default="vscode_issues")
    parser.add_argument("--top-k", type=int, default=5)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    qa_parser = subparsers.add_parser("qa", help="Ask a question about the issues")
    qa_parser.add_argument("--question", required=True)

    triage_parser = subparsers.add_parser("triage", help="Triage a new issue")
    triage_parser.add_argument("--title", required=True)
    triage_parser.add_argument("--body", required=True)

    args = parser.parse_args()
    qdrant, openai_client, embed_model = get_clients(args.collection)

    if args.mode == "qa":
        answer = answer_question(qdrant, openai_client, embed_model, args.collection, args.question, args.top_k)
        print(f"\n--- Answer ---\n{answer}")

    elif args.mode == "triage":
        result = triage_issue(qdrant, openai_client, embed_model, args.collection, args.title, args.body, args.top_k)
        print(f"\n--- Triage result ---\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()