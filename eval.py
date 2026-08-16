"""
RepoTriage — Day 4: Evaluation harness

Runs qa.py's answer_question / triage_issue against a hand-written eval set
(eval_set.json) and checks simple, explicit properties — this is intentionally
NOT "LLM-judges-LLM" scoring; for a project this size, deterministic keyword/
schema checks are more trustworthy and easier to defend in an interview than
another model grading outputs.

Usage:
    python eval.py                  # run full eval set
    python eval.py --mode qa        # only QA cases
    python eval.py --mode triage    # only triage cases
"""

import argparse
import json
import sys
import time

from qa import answer_question, get_clients, triage_issue

COLLECTION_DEFAULT = "vscode_issues"


def run_qa_cases(qdrant, openai_client, embed_model, collection, cases: list[dict]) -> tuple[int, int]:
    passed, total = 0, len(cases)
    for case in cases:
        total_start = time.time()
        answer = answer_question(qdrant, openai_client, embed_model, collection, case["question"])
        elapsed = time.time() - total_start
        answer_lower = answer.lower()

        ok = True
        reason = "ok"

        if case.get("expect_refusal_or_hedge"):
            hedge_markers = ["don't have enough", "not enough information", "does not contain enough",
                      "cannot answer", "context doesn't", "context does not", "no information",
                      "not covered", "unable to answer", "doesn't contain"]
            ok = any(marker in answer_lower for marker in hedge_markers)
            reason = "expected a hedge/refusal for an out-of-domain question, didn't get one" if not ok else "ok"

        elif case.get("expect_keywords_any"):
            ok = any(kw.lower() in answer_lower for kw in case["expect_keywords_any"])
            reason = f"none of expected keywords {case['expect_keywords_any']} found in answer" if not ok else "ok"

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] ({elapsed:.1f}s) QA: {case['question'][:60]!r}")
        if not ok:
            print(f"         reason: {reason}")
            print(f"         answer: {answer[:200]}")

    return passed, total


def run_triage_cases(qdrant, openai_client, embed_model, collection, cases: list[dict]) -> tuple[int, int]:
    passed, total = 0, len(cases)
    for case in cases:
        t0 = time.time()
        result = triage_issue(qdrant, openai_client, embed_model, collection, case["title"], case["body"])
        elapsed = time.time() - t0

        ok = True
        reasons = []

        if "error" in result:
            ok = False
            reasons.append("model did not return valid JSON")
        else:
            if "expect_priority_in" in case and result.get("priority") not in case["expect_priority_in"]:
                ok = False
                reasons.append(f"priority={result.get('priority')!r}, expected one of {case['expect_priority_in']}")
            if "expect_is_duplicate" in case and result.get("is_duplicate") != case["expect_is_duplicate"]:
                ok = False
                reasons.append(f"is_duplicate={result.get('is_duplicate')}, expected {case['expect_is_duplicate']}")

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] ({elapsed:.1f}s) Triage: {case['title'][:60]!r}")
        if not ok:
            print(f"         reasons: {'; '.join(reasons)}")
            print(f"         result: {json.dumps(result)}")

    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Run the RepoTriage eval set")
    parser.add_argument("--collection", default=COLLECTION_DEFAULT)
    parser.add_argument("--eval-set", default="eval_set.json")
    parser.add_argument("--mode", choices=["all", "qa", "triage"], default="all")
    args = parser.parse_args()

    with open(args.eval_set) as f:
        eval_data = json.load(f)

    qdrant, openai_client, embed_model = get_clients(args.collection)

    total_passed, total_cases = 0, 0

    if args.mode in ("all", "qa"):
        print("\n=== QA cases ===")
        p, t = run_qa_cases(qdrant, openai_client, embed_model, args.collection, eval_data["qa_cases"])
        total_passed += p
        total_cases += t

    if args.mode in ("all", "triage"):
        print("\n=== Triage cases ===")
        p, t = run_triage_cases(qdrant, openai_client, embed_model, args.collection, eval_data["triage_cases"])
        total_passed += p
        total_cases += t

    print(f"\n=== Summary: {total_passed}/{total_cases} passed ===")
    if total_passed < total_cases:
        sys.exit(1)  # non-zero exit so this can gate a CI step if you wire it in


if __name__ == "__main__":
    main()
