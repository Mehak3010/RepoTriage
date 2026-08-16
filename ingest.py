"""
RepoTriage — Day 1: Ingest & clean GitHub issues

Pulls issues from a target repo via the GitHub REST API and saves a clean,
structured dataset (Parquet + CSV) ready for embedding in Day 2.

Setup:
    1. Generate a free GitHub personal access token (read-only, "public_repo"
       scope is enough): https://github.com/settings/tokens
    2. export GITHUB_TOKEN="your_token_here"
    3. python ingest.py --repo microsoft/vscode --max-issues 800

Without a token you're capped at 60 requests/hour, which won't get you a
useful dataset. With a token you get 5,000/hour.
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

GITHUB_API = "https://api.github.com"


def get_session(token: str | None) -> requests.Session:
    session = requests.Session()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    session.headers.update(headers)
    return session


def check_rate_limit(session: requests.Session) -> None:
    resp = session.get(f"{GITHUB_API}/rate_limit")
    resp.raise_for_status()
    core = resp.json()["resources"]["core"]
    reset_time = datetime.fromtimestamp(core["reset"], tz=timezone.utc).strftime("%H:%M:%S UTC")
    print(f"Rate limit: {core['remaining']}/{core['limit']} remaining, resets at {reset_time}")
    if core["remaining"] < 20:
        print("WARNING: very low on requests. Consider waiting or adding a GITHUB_TOKEN.")


def fetch_issues(session: requests.Session, repo: str, max_issues: int, state: str = "all") -> list[dict]:
    """
    Fetch issues (GitHub's /issues endpoint includes PRs too — we filter those out).
    Paginates at 100 per page, the API max.
    """
    issues = []
    page = 1
    per_page = 100

    while len(issues) < max_issues:
        url = f"{GITHUB_API}/repos/{repo}/issues"
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc",
        }
        resp = session.get(url, params=params)

        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - int(time.time()), 5)
            MAX_WAIT = 120  # never block for more than 2 minutes — fail loudly instead
            if wait > MAX_WAIT:
                print(f"\nRate limited and reset is {wait}s away — that's too long to wait.")
                print("This almost always means you're running without a GITHUB_TOKEN, or")
                print("you're on a shared/proxied IP that's already used up its quota.")
                print("Set one with: export GITHUB_TOKEN='your_token_here' (see docstring above).")
                sys.exit(1)
            print(f"Rate limited. Sleeping {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break  # no more pages

        # Filter out pull requests (GitHub's issues endpoint mixes them in)
        real_issues = [i for i in batch if "pull_request" not in i]
        issues.extend(real_issues)

        print(f"  page {page}: fetched {len(batch)} items ({len(real_issues)} issues, "
              f"{len(batch) - len(real_issues)} PRs skipped) — total issues so far: {len(issues)}")

        page += 1
        if len(batch) < per_page:
            break  # last page

    return issues[:max_issues]


def clean_to_dataframe(raw_issues: list[dict], repo: str) -> pd.DataFrame:
    rows = []
    for issue in raw_issues:
        rows.append({
            "repo": repo,
            "issue_number": issue["number"],
            "title": issue["title"],
            "body": issue.get("body") or "",
            "state": issue["state"],
            "labels": [label["name"] for label in issue.get("labels", [])],
            "num_comments": issue["comments"],
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "closed_at": issue.get("closed_at"),
            "author": issue["user"]["login"] if issue.get("user") else None,
            "url": issue["html_url"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Basic cleaning
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)
    df["body"] = df["body"].str.strip()
    df["title"] = df["title"].str.strip()

    # Drop issues with essentially no content — not useful for retrieval later
    df = df[(df["title"].str.len() > 0)].reset_index(drop=True)

    # Combined text field we'll embed in Day 2
    df["text_for_embedding"] = df["title"] + "\n\n" + df["body"]

    return df


def main():
    parser = argparse.ArgumentParser(description="Ingest GitHub issues for RepoTriage")
    parser.add_argument("--repo", default="microsoft/vscode",
                         help="owner/repo, e.g. microsoft/vscode")
    parser.add_argument("--max-issues", type=int, default=800)
    parser.add_argument("--state", default="all", choices=["open", "closed", "all"])
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("NOTE: No GITHUB_TOKEN found in environment. You're limited to 60 requests/hour.")
        print("      Set one with: export GITHUB_TOKEN='your_token_here'\n")

    session = get_session(token)

    try:
        check_rate_limit(session)
    except requests.RequestException as e:
        print(f"Could not check rate limit (continuing anyway): {e}")

    print(f"\nFetching issues from {args.repo} (target: {args.max_issues}, state: {args.state})...")
    raw_issues = fetch_issues(session, args.repo, args.max_issues, args.state)
    print(f"\nFetched {len(raw_issues)} raw issues.")

    df = clean_to_dataframe(raw_issues, args.repo)
    print(f"Cleaned dataset: {len(df)} issues after filtering.")

    if df.empty:
        print("No data to save. Check the repo name and your token.")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    parquet_path = os.path.join(args.out_dir, "issues.parquet")
    csv_path = os.path.join(args.out_dir, "issues.csv")

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"\nSaved to:\n  {parquet_path}\n  {csv_path}")
    print("\n--- Quick summary ---")
    print(f"Total issues: {len(df)}")
    print(f"Open: {(df['state'] == 'open').sum()} | Closed: {(df['state'] == 'closed').sum()}")
    print(f"Avg comments per issue: {df['num_comments'].mean():.1f}")
    print(f"Date range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
    top_labels = pd.Series([l for labels in df["labels"] for l in labels]).value_counts().head(5)
    print(f"Top 5 labels:\n{top_labels.to_string()}")


if __name__ == "__main__":
    main()
