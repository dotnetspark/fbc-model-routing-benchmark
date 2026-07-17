"""Lesson 5 analysis — routing behavior + data-referenced recommendation.

Reads results/router_selections.csv (from run_router_dryrun.py) and the measured
Lessons 1-4 results, and produces:
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
    for tier in ("foundation", "instruction_tuned", "slm"):
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

    # --- Chart: 2x2 quadrant "gravity" map per router (small multiples) ---
    # POSITION encodes both axes (x = cheap|strong tier, y = cold|GROUNDED); each
    # router's 45 picks pile into quadrants as bubbles sized by count. The empty
    # GROUNDED (top) half for every real router is the finding, as literal space —
    # not a missing bar segment you have to notice.
    combos = [("cheap", "False", "cheap_ung", "cheap · ungrounded"),
              ("strong", "False", "strong_ung", "strong · ungrounded"),
              ("cheap", "True", "cheap_grd", "cheap · GROUNDED"),
              ("strong", "True", "strong_grd", "strong · GROUNDED")]
    POS = {("cheap", "False"): (0, 0, "cheap_ung"), ("strong", "False"): (1, 0, "strong_ung"),
           ("cheap", "True"): (0, 1, "cheap_grd"), ("strong", "True"): (1, 1, "strong_grd")}
    TXT = {"cheap_ung": INK, "strong_ung": "white", "cheap_grd": "white", "strong_grd": "white"}
    SCALE = 62  # bubble area per question (points^2)

    # Real routers first, custom (the contrast) last; illustrative heuristic flagged.
    ORDER = ["routellm", "notdiamond", "notdiamond_grounded", "notdiamond_trained",
             "difficulty_baseline", "custom"]
    panels = [r for r in ORDER if r in routers] + [r for r in routers if r not in ORDER]
    DISPLAY = {"difficulty_baseline": "difficulty (illustrative)",
               "notdiamond_grounded": "notdiamond (grounded, default)",
               "notdiamond_trained": "notdiamond (grounded, TRAINED)"}

    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.8, 4.4 * nrow), squeeze=False)
    for idx, router in enumerate(panels):
        ax = axes[idx // ncol][idx % ncol]
        rows = [r for r in sel if r["router"] == router]
        ax.axhspan(0.5, 1.9, color="#1baf7a", alpha=0.06)          # the "grounded zone"
        ax.axhline(0.5, color="#d8d8d5", lw=1, zorder=0)
        ax.axvline(0.5, color="#d8d8d5", lw=1, zorder=0)
        for (strength, grd), (x, y, ckey) in POS.items():
            n = sum(r["strength"] == strength and r["grounded"] == grd for r in rows)
            if n:
                ax.scatter([x], [y], s=n * SCALE, color=C[ckey], edgecolor="white",
                           linewidth=1.5, zorder=3)
                ax.text(x, y, str(n), ha="center", va="center", color=TXT[ckey],
                        fontsize=9, weight="bold", zorder=4)
        ax.set_xlim(-0.75, 1.75); ax.set_ylim(-0.75, 1.9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["cheap", "strong"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["cold", "GROUNDED"])
        ax.set_title(DISPLAY.get(router, router), fontsize=11, weight="bold")
        ax.tick_params(length=0)
        ax.spines[["top", "right"]].set_visible(False)
    for j in range(len(panels), nrow * ncol):   # hide any unused panel
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle("Router selection gravity — where each router's 45 picks land", y=0.99, fontsize=13)
    fig.text(0.5, 0.945, "training moves grounded NotDiamond from STRONG (blue) to cheap + GROUNDED (green) — matching the custom router",
             ha="center", fontsize=9, color=MUTED)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C[k], markersize=10)
               for _, _, k, _ in combos]
    fig.legend(handles, [lbl for *_, lbl in combos], fontsize=8, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 0.915), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); plt.savefig(OUT_PNG, dpi=130); print(f"wrote {OUT_PNG}")

    # --- Expected-accuracy overlay: follow each router's picks, score by measured data ---
    print("\nExpected citation accuracy if you followed each router's selections")
    print("(strong→Opus, cheap→Haiku, cheap_local→phi3; grounded per the flag; measured Lessons 1-4):")
    gold = _gold()
    for router in routers:
        tot = corr = 0.0
        for r in (x for x in sel if x["router"] == router):
            _, cat = gold.get(r["question_id"], (None, None))
            # custom routes some picks to the LOCAL phi3 (raw_model "cheap_local+..."),
            # not Haiku — score those as phi3 (slm), else the number is inflated.
            tf = "slm" if r["raw_model"].startswith("cheap_local") else TIER_FILE[r["strength"]]
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
