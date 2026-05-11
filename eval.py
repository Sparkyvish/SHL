"""
eval.py  –  Local evaluation harness.

Usage:
    python eval.py --traces traces/          # run all JSON trace files
    python eval.py --traces traces/t01.json  # single trace

A trace file looks like:
{
  "persona": "...",
  "facts": {"role": "Java developer", "seniority": "mid-level", ...},
  "expected_assessments": ["Java 8 (New)", "OPQ32r", ...]
}

Metrics computed:
  • Recall@10    – fraction of expected assessments found in final shortlist
  • Turn count   – number of turns taken (must be ≤ 8)
  • Schema valid – all responses match the required schema
  • Hallucination rate – recommendations with URLs not in catalog
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000")
TIMEOUT = 30  # seconds per call (matches evaluator cap)

# ── Simulated user ────────────────────────────────────────────────────────────

def _simulate_user_reply(
    agent_reply: str,
    facts: Dict[str, Any],
    turn: int,
    client: "anthropic.Anthropic",  # type: ignore[name-defined]
) -> str:
    """
    Use Claude to simulate a user who knows the facts and answers truthfully.
    Falls back to a heuristic if Anthropic is unavailable.
    """
    try:
        import anthropic
        ac = anthropic.Anthropic()
        resp = ac.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=(
                "You are a hiring manager. Answer the agent's question truthfully "
                "using ONLY the facts below. If you have no relevant fact, say "
                "'I have no preference on that.' Keep answers brief (1-2 sentences).\n\n"
                f"Facts: {json.dumps(facts)}"
            ),
            messages=[{"role": "user", "content": agent_reply}],
        )
        return resp.content[0].text.strip()
    except Exception:
        # Heuristic fallback
        return "I have no strong preference on that."


# ── Recall@K ─────────────────────────────────────────────────────────────────

def recall_at_k(recommended: List[str], expected: List[str], k: int = 10) -> float:
    """Fraction of expected assessments that appear in top-k recommendations."""
    if not expected:
        return 1.0
    top_k_names = {n.lower() for n in recommended[:k]}
    hits = sum(1 for e in expected if e.lower() in top_k_names)
    return hits / len(expected)


# ── Single trace runner ───────────────────────────────────────────────────────

def run_trace(trace: dict, http: httpx.Client) -> dict:
    facts = trace.get("facts", {})
    expected = trace.get("expected_assessments", [])
    persona = trace.get("persona", "anonymous")

    log.info("  Persona: %s | Expected: %s", persona, expected)

    history: List[Dict[str, str]] = []
    final_recs: List[str] = []
    schema_errors = 0
    hallucinated_urls = 0
    catalog_urls = _get_catalog_urls(http)

    # Seed the conversation with the persona's initial message
    initial = facts.get("initial_message", f"I am hiring a {facts.get('role', 'candidate')}.")
    history.append({"role": "user", "content": initial})

    for turn in range(1, 9):  # max 8 turns total
        # ── Call /chat ──────────────────────────────────────────────────────
        try:
            resp = http.post(
                f"{BASE_URL}/chat",
                json={"messages": history},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.error("    Turn %d – /chat call failed: %s", turn, exc)
            schema_errors += 1
            break

        # ── Schema validation ───────────────────────────────────────────────
        for field in ("reply", "recommendations", "end_of_conversation"):
            if field not in data:
                log.warning("    Missing field: %s", field)
                schema_errors += 1

        agent_reply = data.get("reply", "")
        recs = data.get("recommendations", [])
        eoc = data.get("end_of_conversation", False)

        # Hallucination check
        for r in recs:
            if r.get("url") not in catalog_urls:
                hallucinated_urls += 1
                log.warning("    Hallucinated URL: %s", r.get("url"))

        log.info("    Turn %d | recs=%d | eoc=%s", turn, len(recs), eoc)

        history.append({"role": "assistant", "content": agent_reply})

        if recs:
            final_recs = [r["name"] for r in recs]

        if eoc or len(recs) > 0:
            # Conversation done
            break

        if turn >= 8:
            break

        # ── Simulate user response ──────────────────────────────────────────
        user_reply = _simulate_user_reply(agent_reply, facts, turn, None)
        log.info("    Simulated user: %s", user_reply[:80])
        history.append({"role": "user", "content": user_reply})

    recall = recall_at_k(final_recs, expected)
    log.info(
        "  → Recall@10=%.2f | turns=%d | schema_errors=%d | hallucinated_urls=%d",
        recall, len(history), schema_errors, hallucinated_urls,
    )

    return {
        "persona": persona,
        "recall_at_10": recall,
        "turns": len([m for m in history if m["role"] == "user"]),
        "schema_errors": schema_errors,
        "hallucinated_urls": hallucinated_urls,
        "final_recommendations": final_recs,
        "expected_assessments": expected,
    }


def _get_catalog_urls(http: httpx.Client) -> set:
    """Load catalog URLs from local file for hallucination checking."""
    try:
        with open("index_meta.json") as f:
            meta = json.load(f)
        return {a["url"] for a in meta}
    except FileNotFoundError:
        log.warning("index_meta.json not found; skipping URL validation.")
        return set()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SHL Recommender eval harness")
    parser.add_argument(
        "--traces",
        default="traces/",
        help="Path to a trace JSON file or directory of trace files.",
    )
    args = parser.parse_args()

    # Resolve trace files
    if os.path.isdir(args.traces):
        trace_files = sorted(glob.glob(os.path.join(args.traces, "*.json")))
    else:
        trace_files = [args.traces]

    if not trace_files:
        log.error("No trace files found at: %s", args.traces)
        sys.exit(1)

    log.info("Running %d trace(s) against %s", len(trace_files), BASE_URL)

    with httpx.Client() as http:
        # Health check
        try:
            health = http.get(f"{BASE_URL}/health", timeout=120)
            health.raise_for_status()
            log.info("Service is healthy.")
        except Exception as exc:
            log.error("Health check failed: %s", exc)
            sys.exit(1)

        results = []
        for tf in trace_files:
            log.info("Trace: %s", tf)
            with open(tf) as f:
                trace = json.load(f)
            result = run_trace(trace, http)
            results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(results)
    mean_recall = sum(r["recall_at_10"] for r in results) / n
    mean_turns = sum(r["turns"] for r in results) / n
    total_schema_errors = sum(r["schema_errors"] for r in results)
    total_hallucinations = sum(r["hallucinated_urls"] for r in results)

    print("\n" + "=" * 60)
    print(f"Traces evaluated : {n}")
    print(f"Mean Recall@10   : {mean_recall:.3f}")
    print(f"Mean turns       : {mean_turns:.1f}  (max 8)")
    print(f"Schema errors    : {total_schema_errors}")
    print(f"Hallucinated URLs: {total_hallucinations}")
    print("=" * 60)

    # Dump detailed results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("Detailed results saved → eval_results.json")


if __name__ == "__main__":
    main()
