"""Lesson 5 dry-run driver.

Runs each router's SELECTION over the 45 evaluation questions — which model would
it pick, and would it ground — WITHOUT calling the target LLM. Writes
results/router_selections.csv for the analysis.

    python run_router_dryrun.py                 # all routers
    python run_router_dryrun.py --skip notdiamond   # e.g. when NOT_DIAMOND_API_KEY isn't set

NotDiamond needs NOT_DIAMOND_API_KEY (note the underscores); custom and
difficulty_baseline need nothing.
"""

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

from routers import custom, difficulty_baseline, notdiamond_router, notdiamond_grounded

load_dotenv()

ROUTERS = {
    "custom": custom.select,
    "difficulty_baseline": difficulty_baseline.select,
    "notdiamond": notdiamond_router.select,
    # NotDiamond fed the GROUNDED prompt — "compose retrieval + off-the-shelf router"
    # instead of building a bespoke one. See routers/notdiamond_grounded.py.
    "notdiamond_grounded": notdiamond_grounded.select,
}

# RouteLLM's ML stack only installs on Linux (the Docker image); include it when
# importable, skip it silently on the Windows host. See docker/Dockerfile.
try:
    from routers import routellm_router
    ROUTERS["routellm"] = routellm_router.select
except Exception:
    pass
DATA = Path("data/fbc_eval_questions.csv")
CONTEXT = Path("data/fbc_eval_context.csv")   # retrieved passages for the grounded variant
OUT = Path("results/router_selections.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip", nargs="*", default=[], help="router names to skip")
    args = ap.parse_args()

    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    contexts = ({r["question_id"]: r["context"] for r in csv.DictReader(CONTEXT.open(encoding="utf-8"))}
                if CONTEXT.exists() else {})
    routers = {k: v for k, v in ROUTERS.items() if k not in args.skip}
    print(f"Routers: {list(routers)}  |  {len(rows)} questions  |  {len(contexts)} contexts")

    out = []
    for r in rows:
        for name, fn in routers.items():
            try:
                c = fn(r["question"], r["category"], context=contexts.get(r["question_id"]))
                if c is None:      # router abstained for this question (e.g. grounded router, ungroundable q)
                    continue
                out.append({"question_id": r["question_id"], "category": r["category"],
                            "router": name, "strength": c.strength,
                            "grounded": c.grounded, "raw_model": c.raw_model})
            except Exception as e:  # a router failing is data, not a crash
                out.append({"question_id": r["question_id"], "category": r["category"],
                            "router": name, "strength": "ERROR", "grounded": None,
                            "raw_model": f"{type(e).__name__}: {e}"[:100]})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_id", "category", "router",
                                          "strength", "grounded", "raw_model"])
        w.writeheader(); w.writerows(out)
    print(f"Wrote {len(out)} selections -> {OUT}")

    # Quick gravity summary per router.
    for name in routers:
        sel = [r for r in out if r["router"] == name and r["strength"] != "ERROR"]
        errs = sum(1 for r in out if r["router"] == name and r["strength"] == "ERROR")
        if not sel:
            print(f"  {name:20} ALL ERRORED ({errs})"); continue
        strong = sum(r["strength"] == "strong" for r in sel)
        grounded = sum(bool(r["grounded"]) for r in sel)
        print(f"  {name:20} strong {strong/len(sel):5.0%} | grounded {grounded/len(sel):5.0%}"
              f"  (n={len(sel)}, errors={errs})")


if __name__ == "__main__":
    main()
