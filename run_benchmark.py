"""
Cold-run benchmark harness (Lesson 1, reused by Lessons 2-4).

Loads the evaluation set, fans each question out to a model client N times with
no retrieved context, and writes one JSON line per request to
results/<model_class>_raw.jsonl. The analysis notebook
(analysis/benchmark_report.ipynb) consumes that file — this script is the only
place inference happens, so there is exactly one source of truth for how a
number was produced.

Run from the repo root so the `clients.` imports resolve:

    python run_benchmark.py                      # foundation client, 3 repeats per question
    python run_benchmark.py --repeats 1 --limit 2    # cheap smoke test (2 questions, 1 run each)
    python run_benchmark.py --concurrency 8          # more parallel requests

Later lessons register their client function in CLIENTS and pass
--model-class instruction_tuned / slm / multimodal.
"""

import argparse
import asyncio
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from clients.citation_utils import extract_section_citation, section_present_in_context
from clients import foundation_client, instruction_tuned_client, slm_client

# One entry per model class: (client function, exceptions to record as failed
# requests). Lesson 4+ clients get registered here with their own error types.
CLIENTS = {
    "foundation": (foundation_client.call_foundation_model, foundation_client.API_ERRORS),
    "instruction_tuned": (instruction_tuned_client.call_instruction_tuned,
                          instruction_tuned_client.API_ERRORS),
    "slm": (slm_client.call_slm, slm_client.API_ERRORS),
}

# Model-name constant per class — used by --resume to keep only prior successes
# from the CURRENT model (a model swap should re-run, not reuse stale rows).
MODEL_NAMES = {
    "foundation": foundation_client.FOUNDATION_MODEL,
    "instruction_tuned": instruction_tuned_client.INSTRUCTION_TUNED_MODEL,
    "slm": slm_client.SLM_MODEL,
}

# The $0 path: Gemini's free tier covers the whole benchmark (Lesson 1 provider
# note). Registered lazily — the module builds its genai.Client at import time,
# which requires GEMINI_API_KEY, and a missing key must not break the default
# Anthropic/Ollama classes.
try:
    from clients import foundation_client_gemini
    CLIENTS["foundation_gemini"] = (foundation_client_gemini.call_foundation_model,
                                    foundation_client_gemini.API_ERRORS)
    MODEL_NAMES["foundation_gemini"] = foundation_client_gemini.FOUNDATION_MODEL
except Exception:
    pass  # no GEMINI_API_KEY in the environment — the free-tier class is optional

DATA_PATH = Path("data/fbc_eval_questions.csv")
CONTEXT_PATH = Path("data/fbc_eval_context.csv")
RESULTS_DIR = Path("results")


def load_eval_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_contexts() -> dict:
    """question_id -> retrieved code passage (Lesson 6 grounding). Questions with
    no passage (e.g. q017, a FEMA date not in any code) are simply absent."""
    if not CONTEXT_PATH.exists():
        return {}
    with CONTEXT_PATH.open(encoding="utf-8", newline="") as f:
        return {r["question_id"]: r["context"] for r in csv.DictReader(f)}


def build_grounded_prompt(question: str, context: str) -> str:
    """Lesson 6 inference-time contract: answer only from the supplied passage,
    cite the section as it appears there, and admit when the answer isn't present."""
    return (
        "Answer the Florida Building Code question using ONLY the context below.\n"
        'If the answer is not in the context, say "Not found in context."\n'
        "Always cite the section number exactly as it appears in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
    )


async def run_one(
    sem: asyncio.Semaphore,
    client_fn,
    api_errors: tuple,
    model_class: str,
    row: dict,
    repeat: int,
    context: str | None = None,
) -> dict:
    """Run one (question, repeat) pair; never raises on API errors — failures
    become rows too, because the failure rate is itself a rubric metric.

    When `context` is given (Lesson 6 grounded run), the question is wrapped with
    the retrieved passage and the question_id is suffixed `_grounded`."""
    grounded = context is not None
    qid = f"{row['question_id']}_grounded" if grounded else row["question_id"]
    prompt = build_grounded_prompt(row["question"], context) if grounded else row["question"]
    async with sem:
        start = time.perf_counter()
        try:
            result = await client_fn(prompt, qid)
            record = asdict(result)
            record["error"] = None
        except api_errors as e:
            record = {
                "model_class": model_class,
                "model_name": None,
                "question_id": qid,
                "output": None,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "cited_section": None,
                "error": f"{type(e).__name__}: {e}",
            }

    # Gold sections are prose-styled ("FBC Building 1020.3"); cited_section is
    # already normalized ("1020.3"). Normalize gold with the same extractor so
    # the match is apples-to-apples.
    gold_norm = extract_section_citation(row["gold_section"])
    record["repeat"] = repeat
    record["category"] = row["category"]
    record["gold_section"] = row["gold_section"]
    record["gold_section_norm"] = gold_norm
    record["citation_match"] = (
        record["error"] is None
        and record["cited_section"] is not None
        and gold_norm is not None
        and record["cited_section"] == gold_norm
    )
    # Grounding-compliance (Lesson 6): did the model cite a section that is
    # ACTUALLY in the supplied passage, or invent a plausible one anyway? Only
    # meaningful for grounded runs; None otherwise.
    record["grounding_compliant"] = (
        section_present_in_context(record["cited_section"], context) if grounded else None
    )
    return record


def print_summary(records: list[dict], model_class: str) -> None:
    ok = [r for r in records if r["error"] is None]
    failures = [r for r in records if r["error"] is not None]

    print(f"\n=== {model_class}: {len(records)} runs "
          f"({len(ok)} succeeded, {len(failures)} failed) ===")

    if failures:
        print(f"Failure rate (reliability metric): {len(failures) / len(records):.1%}")
        for r in failures[:5]:
            print(f"  {r['question_id']} (repeat {r['repeat']}): {r['error']}")

    unmatchable = {r["question_id"] for r in records if r["gold_section_norm"] is None}
    if unmatchable:
        print(f"\nNOTE: {len(unmatchable)} question(s) have a gold_section the regex "
              f"extractor can't normalize ({', '.join(sorted(unmatchable))}) — these can "
              f"never score a citation match and inflate the hallucination rate. "
              f"Review them or score them separately.")

    if not ok:
        return

    # Scoreable = succeeded AND the gold section normalizes. Within those,
    # distinguish citing the WRONG section (hallucination per the rubric)
    # from giving NO citation (abstention — a different, more honest failure).
    scoreable = [r for r in ok if r["gold_section_norm"] is not None]
    matched = [r for r in scoreable if r["citation_match"]]
    no_cite = [r for r in scoreable if r["cited_section"] is None]
    wrong = [r for r in scoreable if r["cited_section"] is not None and not r["citation_match"]]
    n = len(scoreable)
    print(f"\nOf {n} scoreable runs: "
          f"{len(matched)} matched ({len(matched)/n:.1%}) | "
          f"{len(wrong)} wrong citation ({len(wrong)/n:.1%}) | "
          f"{len(no_cite)} no citation ({len(no_cite)/n:.1%})")
    print(f"Hallucination rate (wrong/invented citation): {len(wrong)/n:.1%}   <- headline number")
    print(f"Citation-match rate:                          {len(matched)/n:.1%}")

    # Grounded runs: did the cited section actually appear in the supplied passage?
    grounded = [r for r in ok if r.get("grounding_compliant") is not None]
    if grounded:
        compliant = sum(r["grounding_compliant"] for r in grounded)
        print(f"Grounding compliance (cited a section IN the context): {compliant/len(grounded):.1%}"
              f"  ({compliant}/{len(grounded)})")

    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r["citation_match"])
    print("\nCitation match by category:")
    for cat, matches in sorted(by_cat.items()):
        print(f"  {cat:25} {sum(matches) / len(matches):6.1%}   (n={len(matches)})")

    latencies = sorted(r["latency_ms"] for r in ok)
    p95 = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)]
    total_cost = sum(r["cost_usd"] for r in ok)
    print(f"\nLatency p50 / p95: {statistics.median(latencies):.0f} ms / {p95:.0f} ms")
    print(f"Total run cost:    ${total_cost:.4f} "
          f"(${total_cost / len(ok):.5f} per request)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--model-class", choices=CLIENTS, default="foundation")
    parser.add_argument("--repeats", type=int, default=3,
                        help="runs per question (default 3, per Lesson 1)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="max in-flight API requests (default 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N questions (smoke tests)")
    parser.add_argument("--output", type=Path, default=None,
                        help="override results/<model_class>_raw.jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="keep successful rows already in the output file and "
                             "only run the missing/failed (question, repeat) pairs — "
                             "essential on free tiers with small daily quotas")
    parser.add_argument("--grounded", action="store_true",
                        help="Lesson 6: inject the retrieved code passage per question "
                             "(from data/fbc_eval_context.csv); writes "
                             "results/<class>_grounded.jsonl and skips questions with no passage")
    args = parser.parse_args()

    rows = load_eval_set(DATA_PATH)
    if args.limit:
        rows = rows[: args.limit]

    contexts = load_contexts() if args.grounded else {}
    if args.grounded:
        rows = [r for r in rows if r["question_id"] in contexts]
        print(f"GROUNDED mode: {len(rows)} questions have a retrieved context passage")

    client_fn, api_errors = CLIENTS[args.model_class]
    suffix = "grounded" if args.grounded else "raw"
    out_path = args.output or RESULTS_DIR / f"{args.model_class}_{suffix}.jsonl"

    # Resume: keep prior successes for the CURRENT model only (rows from a
    # different model or failed rows get re-run), so the final file is a clean
    # single-model run even when assembled across several days/quota windows.
    kept: list[dict] = []
    done: set[tuple] = set()
    if args.resume and out_path.exists():
        current_model = MODEL_NAMES[args.model_class]
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("error") is None and r.get("model_name") == current_model:
                kept.append(r)
                done.add((r["question_id"], r["repeat"]))
        print(f"Resume: keeping {len(kept)} successful rows, "
              f"re-running the rest.")

    sem = asyncio.Semaphore(args.concurrency)
    # done stores the AS-WRITTEN question_id (suffixed `_grounded` in grounded runs),
    # so build the effective id per row when checking what still needs running.
    def eff_qid(row):
        return f"{row['question_id']}_grounded" if args.grounded else row["question_id"]
    tasks = [
        run_one(sem, client_fn, api_errors, args.model_class, row, repeat,
                context=contexts.get(row["question_id"]) if args.grounded else None)
        for row in rows
        for repeat in range(1, args.repeats + 1)
        if (eff_qid(row), repeat) not in done
    ]
    print(f"Running {len(tasks)} requests "
          f"({len(rows)} questions x {args.repeats} repeats, "
          f"concurrency {args.concurrency})...")
    new_records = await asyncio.gather(*tasks)

    records = sorted(kept + list(new_records),
                     key=lambda r: (r["question_id"], r["repeat"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows -> {out_path}")

    print_summary(records, args.model_class)


if __name__ == "__main__":
    asyncio.run(main())
