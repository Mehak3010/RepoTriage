"""
RepoTriage — Day 2: Embed issues + upload to Qdrant

Embeds each issue's title+body using a free local model (no API key, no
per-call cost) and uploads the vectors + metadata to a Qdrant collection
for semantic search.

Setup — Qdrant Cloud (recommended, no Docker needed):
    1. Sign up free at https://cloud.qdrant.io
    2. Create a free cluster (1GB, plenty for this project)
    3. Copy your cluster URL and API key
    4. export QDRANT_URL="https://xxxx.cloud.qdrant.io"
       export QDRANT_API_KEY="your_key_here"

Setup — Local Qdrant via Docker (alternative):
    1. docker run -p 6333:6333 qdrant/qdrant
    2. export QDRANT_URL="http://localhost:6333"
       (no API key needed for local)

Usage:
    python embed.py --input data/issues.parquet --collection vscode_issues
"""

import argparse
import os
import sys
import time

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "all-MiniLM-L6-v2"  
EMBEDDING_DIM = 384
BATCH_SIZE = 64


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run ingest.py first (Day 1).")
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} issues from {path}")
    return df


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    print(f"Embedding {len(texts)} texts with {MODEL_NAME}...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"Done in {time.time() - t0:.1f}s")
    return embeddings.tolist()


def get_qdrant_client() -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")  # None is fine for local Qdrant

    if not url:
        print("ERROR: QDRANT_URL not set.")
        print("  Qdrant Cloud: export QDRANT_URL='https://xxxx.cloud.qdrant.io'")
        print("  Local Docker: export QDRANT_URL='http://localhost:6333'")
        sys.exit(1)

    return QdrantClient(url=url, api_key=api_key)


def create_collection(client: QdrantClient, collection_name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists — will upsert into it.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{collection_name}'")


def upload_points(client: QdrantClient, collection_name: str, df: pd.DataFrame, embeddings: list[list[float]]) -> None:
    points = []
    for i, (_, row) in enumerate(df.iterrows()):
        points.append(
            PointStruct(
                id=int(row["issue_number"]),  # issue numbers are unique per repo, good as IDs
                vector=embeddings[i],
                payload={
                    "issue_number": int(row["issue_number"]),
                    "title": row["title"],
                    "body": row["body"][:2000],  # cap payload size
                    "state": row["state"],
                    "labels": row["labels"].tolist() if hasattr(row["labels"], "tolist") else list(row["labels"]),
                    "num_comments": int(row["num_comments"]),
                    "created_at": str(row["created_at"]),
                    "url": row["url"],
                },
            )
        )

    print(f"Uploading {len(points)} points to Qdrant...")
    # Upload in batches to avoid oversized requests
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i + BATCH_SIZE]
        client.upsert(collection_name=collection_name, points=batch)
        print(f"  uploaded {min(i + BATCH_SIZE, len(points))}/{len(points)}")

    print("Upload complete.")


def test_search(client: QdrantClient, model: SentenceTransformer, collection_name: str) -> None:
    """Sanity check: run one semantic query and print top results."""
    test_query = "crash on startup"
    query_vector = model.encode(test_query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=5,
    ).points

    print(f"\n--- Test search: '{test_query}' ---")
    for r in results:
        print(f"  [{r.score:.3f}] #{r.payload['issue_number']}: {r.payload['title'][:80]}")


def main():
    parser = argparse.ArgumentParser(description="Embed issues and upload to Qdrant")
    parser.add_argument("--input", default="data/issues.parquet")
    parser.add_argument("--collection", default="repo_issues")
    args = parser.parse_args()

    df = load_data(args.input)

    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = embed_texts(model, df["text_for_embedding"].tolist())

    client = get_qdrant_client()
    create_collection(client, args.collection)
    upload_points(client, args.collection, df, embeddings)

    test_search(client, model, args.collection)

    print(f"\nDone. Collection '{args.collection}' has {len(df)} issues ready for semantic search.")


if __name__ == "__main__":
    main()
