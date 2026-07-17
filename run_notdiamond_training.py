"""Lesson 7 — train a CUSTOM NotDiamond router on grounded prompts.

The hypothesis (from the reader's own push): instead of hand-building a router,
can we *train* NotDiamond so it learns to route grounded prompts to the CHEAP tier
— matching the custom router — where the DEFAULT router instead goes strong?

Pipeline (run in the Docker image, which has notdiamond + anthropic):
  1. Build a training set: run both candidate models on the 44 grounded prompts and
     score each with our citation-match metric.
  2. Write it in NotDiamond's required CSV format (see TRAINING_CSV_FORMAT below).
  3. train_custom_router -> preference_id (trains async on NotDiamond's cloud).
  4. Poll until the trained router is live.
  5. Re-route the 44 grounded prompts with the trained router (tradeoff="cost"), so a
     router that learned "haiku is ~as good as sonnet when grounded" can pick cheap.
  6. Append the picks as router "notdiamond_trained" to results/router_selections.csv.

    docker compose run --rm routers python run_notdiamond_training.py

Needs ANTHROPIC_API_KEY (candidate responses) and NOTDIAMOND_API_KEY (train + route),
both loaded from .env via compose's env_file.

------------------------------------------------------------------------------------
NotDiamond training CSV format (verified against the notdiamond 1.7.0 SDK docstring):
  - one PROMPT column (its name is passed as prompt_column="prompt")
  - for EACH candidate model, two columns named EXACTLY:
        {provider}/{model}/score       (float; higher = better, since maximize=True)
        {provider}/{model}/response     (the model's raw answer text)
  - >= 25 rows (we have 44 groundable questions)
Example header:
  prompt,anthropic/claude-3-7-sonnet-latest/score,anthropic/claude-3-7-sonnet-latest/response,anthropic/claude-3-haiku-20240307/score,anthropic/claude-3-haiku-20240307/response
------------------------------------------------------------------------------------
"""

import csv
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clients.citation_utils import extract_section_citation

load_dotenv()

QUESTIONS = Path("data/fbc_eval_questions.csv")
CONTEXT = Path("data/fbc_eval_context.csv")
TRAIN_CSV = Path("results/notdiamond_training_set.csv")
SELECTIONS = Path("results/router_selections.csv")
PREF_FILE = Path("results/notdiamond_preference_id.txt")

# Candidate models: must be callable by ANTHROPIC_API_KEY *and* in NotDiamond's
# catalog (the 3.x models the key can't call, and 4-7/4-8 aren't in ND's catalog —
# 4-5/4-6 are the overlap). cheap = claude-haiku-4-5 (the benchmark's own Haiku tier,
# so cheap picks map exactly to our measured Haiku-grounded accuracy).
STRONG = ("anthropic", "claude-sonnet-4-5")
CHEAP = ("anthropic", "claude-haiku-4-5-20251001")
CANDIDATES = [STRONG, CHEAP]
PROVIDERS = [{"provider": p, "model": m} for (p, m) in CANDIDATES]

PROMPT_TMPL = ("Answer using ONLY the building-code excerpt below. Cite the section.\n\n"
               "EXCERPT:\n{context}\n\nQUESTION: {question}")


def _col(provider, model, kind):
    return f"{provider}/{model}/{kind}"        # NotDiamond's required naming


def build_dataset():
    """Call each candidate on every grounded prompt; score citation-match. Returns
    in-memory rows (incl. question_id/category for later re-routing)."""
    import anthropic
    ac = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    questions = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    contexts = {r["question_id"]: r["context"] for r in csv.DictReader(CONTEXT.open(encoding="utf-8"))}
    gold = {r["question_id"]: extract_section_citation(r["gold_section"]) for r in questions}

    rows = []
    for q in questions:
        ctx = contexts.get(q["question_id"])
        if not ctx:
            continue  # ungroundable (q017 — a date, no code section)
        prompt = PROMPT_TMPL.format(context=ctx, question=q["question"])
        row = {"prompt": prompt, "_qid": q["question_id"], "_cat": q["category"]}
        marks = []
        for (prov, model) in CANDIDATES:
            resp = ac.messages.create(model=model, max_tokens=500,
                                      messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            cited = extract_section_citation(text)
            score = 1.0 if (cited and cited == gold[q["question_id"]]) else 0.0
            row[_col(prov, model, "response")] = text
            row[_col(prov, model, "score")] = score
            marks.append(f"{model.split('-')[1]}={score:.0f}")
        rows.append(row)
        print(f"  {q['question_id']}: " + "  ".join(marks))
    return rows


def write_training_csv(rows):
    """Write ONLY NotDiamond's required columns (extra cols can fail validation)."""
    cols = ["prompt"] + [_col(p, m, k) for (p, m) in CANDIDATES for k in ("score", "response")]
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    with TRAIN_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"Wrote {len(rows)} rows -> {TRAIN_CSV}  (cols: {cols})")


def train(client):
    print("Uploading training set + training on NotDiamond's cloud...")
    with TRAIN_CSV.open("rb") as f:
        resp = client.custom_router.train_custom_router(
            dataset_file=f,
            language="english",
            llm_providers=json.dumps(PROVIDERS),
            maximize=True,               # higher citation-match score is better
            prompt_column="prompt",
        )
    print("train response:", repr(resp)[:300])
    pref = getattr(resp, "preference_id", None) or getattr(resp, "id", None)
    if pref:
        PREF_FILE.write_text(str(pref), encoding="utf-8")
    print("preference_id:", pref)
    return pref


def wait_until_ready(client, pref, probe_prompt, timeout=1500):
    """Poll: the trained router is usable once select_model(preference_id=...) stops erroring."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            client.model_router.select_model(
                llm_providers=PROVIDERS, messages=[{"role": "user", "content": probe_prompt}],
                preference_id=pref, tradeoff="cost")
            print(f"  trained router live after {int(time.time()-start)}s")
            return True
        except Exception as e:
            print(f"  training... {int(time.time()-start)}s: {str(e)[:90]}")
            time.sleep(30)
    return False


def reroute(client, rows, pref):
    """Route each grounded prompt with the TRAINED router; record tier picks."""
    out = []
    for r in rows:
        resp = client.model_router.select_model(
            llm_providers=PROVIDERS, messages=[{"role": "user", "content": r["prompt"]}],
            preference_id=pref, tradeoff="cost")
        top = resp.providers[0]
        strength = "strong" if top.model == STRONG[1] else "cheap"
        out.append({"question_id": r["_qid"], "category": r["_cat"],
                    "router": "notdiamond_trained", "strength": strength,
                    "grounded": True, "raw_model": f"{top.provider}/{top.model}"})
    return out


def append_selections(new_rows):
    """Merge notdiamond_trained rows into results/router_selections.csv (idempotent)."""
    cols = ["question_id", "category", "router", "strength", "grounded", "raw_model"]
    existing = []
    if SELECTIONS.exists():
        existing = [r for r in csv.DictReader(SELECTIONS.open(encoding="utf-8"))
                    if r["router"] != "notdiamond_trained"]
    with SELECTIONS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(existing)
        w.writerows(new_rows)
    strong = sum(r["strength"] == "strong" for r in new_rows)
    print(f"notdiamond_trained: strong {strong}/{len(new_rows)} | grounded {len(new_rows)}/{len(new_rows)} "
          f"-> appended to {SELECTIONS}")


def _grounded_prompts():
    """Rebuild the grounded prompts from CSVs WITHOUT calling any candidate models —
    for re-routing an already-trained router (routing calls only, no inference cost)."""
    questions = list(csv.DictReader(QUESTIONS.open(encoding="utf-8")))
    contexts = {r["question_id"]: r["context"] for r in csv.DictReader(CONTEXT.open(encoding="utf-8"))}
    out = []
    for q in questions:
        ctx = contexts.get(q["question_id"])
        if ctx:
            out.append({"_qid": q["question_id"], "_cat": q["category"],
                        "prompt": PROMPT_TMPL.format(context=ctx, question=q["question"])})
    return out


def reroute_only():
    """Re-route with the saved trained router (no retrain, no candidate calls). Use this
    after training has had time to finish, since select_model may fall back to the default
    router while training is still in progress."""
    import os
    from notdiamond import NotDiamond
    client = NotDiamond(api_key=os.environ.get("NOTDIAMOND_API_KEY") or os.environ.get("NOT_DIAMOND_API_KEY"))
    pref = PREF_FILE.read_text(encoding="utf-8").strip()
    print(f"Re-routing with trained router preference_id={pref}")
    append_selections(reroute(client, _grounded_prompts(), pref))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--reroute-only":
        reroute_only()
        return

    from notdiamond import NotDiamond
    import os
    client = NotDiamond(api_key=os.environ.get("NOTDIAMOND_API_KEY") or os.environ.get("NOT_DIAMOND_API_KEY"))

    rows = build_dataset()
    if len(rows) < 25:
        raise SystemExit(f"Only {len(rows)} samples — NotDiamond needs >= 25.")
    write_training_csv(rows)

    pref = train(client)
    if not pref:
        raise SystemExit("No preference_id returned — check the train response above.")
    if not wait_until_ready(client, pref, rows[0]["prompt"]):
        raise SystemExit("Trained router did not become ready within the timeout.")

    append_selections(reroute(client, rows, pref))
    print("\nDone. Re-run analysis/router_comparison.py to add the notdiamond_trained panel.")


if __name__ == "__main__":
    main()
