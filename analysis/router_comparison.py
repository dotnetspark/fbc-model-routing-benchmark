"""Lesson 7 analysis — routing behavior + data-referenced recommendation.

Reads results/router_selections.csv (from run_router_dryrun.py) and the measured
Lessons 1-6 results, and produces:
  1. The selection-gravity chart: strength x grounding per router. The payoff is
     visual — the grounded band is empty for every router but the custom one.
  2. An expected-accuracy overlay: if you actually followed each router's picks,
     what citation accuracy would you get, per the measured data? (Descriptive
     judgment, not a self-scored contest — see the caveats printed at the end.)

Run in the 3.11 venv:  .venv311/Scripts/python analysis/router_comparison.py
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console is cp1252 by default
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent
SEL = ROOT / "results" / "router_selections.csv"
OUT_PNG = ROOT / "results" / "router_selection_gravity.png"

# Palette (matches the notebook's validated set).
C = {"cheap_ung": "#adb5bd", "strong_ung": "#52514e",   # ungrounded: grays
     "cheap_grd": "#1baf7a", "strong_grd": "#2a78d6"}    # grounded: aqua / blue
INK, MUTED = "#0b0b0b", "#898781"

# strength tier -> which measured benchmark tier stands in for it (stated assumption:
# a generic router's "cheap" pick ~ our cheap hosted tier Haiku; "strong" ~ Opus).
TIER_FILE = {"strong": "foundation", "cheap": "instruction_tuned"}


def _gold():
    import sys
    sys.path.insert(0, str(ROOT))
    from clients.citation_utils import extract_section_citation
    return {r["question_id"]: (extract_section_citation(r["gold_section"]), r["category"])
            for r in csv.DictReader((ROOT / "data/fbc_eval_questions.csv").open(encoding="utf-8"))}


def measured_accuracy():
    """acc[(tier_file, grounded)][category] = correct-rate, re-scored with current gold."""
    gold = _gold()
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [correct, total]
    for tier in ("foundation", "instruction_tuned"):
        for grounded, suffix in ((False, "raw"), (True, "grounded")):
            p = ROOT / "results" / f"{tier}_{suffix}.jsonl"
            if not p.exists():
                continue
            for line in p.open(encoding="utf-8"):
                r = json.loads(line)
                if r["error"] is not None:
                    continue
                base = r["question_id"].replace("_grounded", "")
                g, cat = gold.get(base, (None, None))
                if not g:
                    continue
                cell = acc[(tier, grounded)][cat]
                cell[1] += 1
                cell[0] += int(r["cited_section"] == g)
    return {k: {c: (v[0] / v[1] if v[1] else 0.0) for c, v in d.items()} for k, d in acc.items()}


def main():
    if not SEL.exists():
        raise SystemExit(f"{SEL} not found — run run_router_dryrun.py first")
    sel = [r for r in csv.DictReader(SEL.open(encoding="utf-8")) if r["strength"] != "ERROR"]
    routers = sorted({r["router"] for r in sel})
    acc = measured_accuracy()

    # --- Chart: stacked horizontal bar per router over the 4 combos ---
    combos = [("cheap", "False", "cheap_ung", "cheap · ungrounded"),
              ("strong", "False", "strong_ung", "strong · ungrounded"),
              ("cheap", "True", "cheap_grd", "cheap · GROUNDED"),
              ("strong", "True", "strong_grd", "strong · GROUNDED")]
    fig, ax = plt.subplots(figsize=(8.5, 0.9 + 0.7 * len(routers)))
    for i, router in enumerate(routers):
        rows = [r for r in sel if r["router"] == router]
        left = 0
        for strength, grd, ckey, _ in combos:
            n = sum(r["strength"] == strength and r["grounded"] == grd for r in rows)
            if n:
                ax.barh(i, n, left=left, color=C[ckey], edgecolor="white", linewidth=1.2)
                ax.text(left + n / 2, i, str(n), ha="center", va="center", color="white", fontsize=8, weight="bold")
            left += n
    ax.set_yticks(range(len(routers))); ax.set_yticklabels(routers)
    ax.set_xlabel("number of questions"); ax.set_title("Router selection gravity — strength × grounding")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for _, _, k, _ in combos]
    ax.legend(handles, [lbl for *_, lbl in combos], fontsize=8, ncol=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=120, bbox_inches="tight"); print(f"wrote {OUT_PNG}")

    # --- Expected-accuracy overlay: follow each router's picks, score by measured data ---
    print("\nExpected citation accuracy if you followed each router's selections")
    print("(strong→Opus, cheap→Haiku; grounded per the flag; measured Lessons 1-6):")
    gold = _gold()
    for router in routers:
        tot = corr = 0.0
        for r in (x for x in sel if x["router"] == router):
            _, cat = gold.get(r["question_id"], (None, None))
            tf = TIER_FILE[r["strength"]]
            a = acc.get((tf, r["grounded"] == "True"), {}).get(cat)
            if a is not None:
                corr += a; tot += 1
        print(f"  {router:20} ~{corr/tot:5.1%}  (over {int(tot)} scoreable picks)")

    print("\nCaveats: descriptive study of SELECTION behavior, not a held-out contest. "
          "The custom router encodes the measured findings by design; the value is the "
          "empty grounded band for generic routers (they cannot express grounding) plus "
          "the expected-accuracy gap. Small n; strength-tier leveling; the strong≈Opus / "
          "cheap≈Haiku mapping is a stated assumption for the overlay.")


if __name__ == "__main__":
    main()
