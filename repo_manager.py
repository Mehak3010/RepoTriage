"""
RepoTriage -- repo_manager.py

Wraps ingest.py + embed.py into a single callable so the API can add a new
repo on demand instead of only ever serving the one repo baked into .env.

Collection naming: "owner/name" -> "owner_name" (Qdrant collection names
can't contain slashes).
"""

import os
import re
import time

import pandas as pd
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from embed import create_collection, embed_texts, upload_points
from ingest import check_rate_limit, clean_to_dataframe, fetch_issues, get_session


def repo_to_collection_name(repo: str) -> str:
    """'microsoft/vscode' -> 'microsoft_vscode'. Qdrant collection names are
    restricted to alphanumeric/underscore/hyphen, so sanitize defensively."""
    slug = repo.replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_-]", "_", slug).lower()


def list_repos(qdrant: QdrantClient) -> list[dict]:
    """Return metadata for every repo (collection) currently ingested."""
    collections = qdrant.get_collections().collections
    result = []
    for c in collections:
        info = qdrant.get_collection(c.name)
        result.append({
            "collection": c.name,
            "issue_count": info.points_count,
        })
    return result


def add_repo(
    repo: str,
    max_issues: int,
    embed_model: SentenceTransformer,
    qdrant: QdrantClient,
    progress_cb=None,
) -> dict:
    """
    Full pipeline for one repo: fetch issues from GitHub -> clean -> embed ->
    upload to a new (or existing) Qdrant collection.

    progress_cb, if given, is called with short status strings -- used by
    app.py to update an in-memory job status for polling.
    """
    def report(msg: str):
        if progress_cb:
            progress_cb(msg)
        print(msg)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set -- required to fetch issues from GitHub's API.")

    collection_name = repo_to_collection_name(repo)

    report(f"Fetching issues from {repo}...")
    session = get_session(token)
    try:
        check_rate_limit(session)
    except Exception:
        pass  # non-fatal, just a courtesy log

    raw_issues = fetch_issues(session, repo, max_issues, state="all")
    if not raw_issues:
        raise RuntimeError(f"No issues found for '{repo}'. Check the repo name (owner/name) and that it's public.")

    report(f"Fetched {len(raw_issues)} issues. Cleaning...")
    df = clean_to_dataframe(raw_issues, repo)
    if df.empty:
        raise RuntimeError(f"All fetched issues for '{repo}' were filtered out (empty titles).")

    # Save alongside the original Day 1 dataset, namespaced by repo so multiple
    # repos' raw data can coexist without overwriting each other.
    os.makedirs("data/repos", exist_ok=True)
    parquet_path = f"data/repos/{collection_name}.parquet"
    df.to_parquet(parquet_path, index=False)

    report(f"Embedding {len(df)} issues (this can take a minute)...")
    t0 = time.time()
    embeddings = embed_texts(embed_model, df["text_for_embedding"].tolist())
    report(f"Embedded in {time.time() - t0:.1f}s. Uploading to Qdrant...")

    create_collection(qdrant, collection_name)
    upload_points(qdrant, collection_name, df, embeddings)

    report(f"Done -- '{repo}' ready as collection '{collection_name}' ({len(df)} issues).")

    return {
        "repo": repo,
        "collection": collection_name,
        "issue_count": len(df),
    }