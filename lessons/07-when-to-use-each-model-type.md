# Lesson 7 — When to Use Each Model Type (Decision Engine + Dashboard)

## 1. Concept

Model selection is a constrained-optimization problem: given a latency budget, a cost ceiling, and an accuracy floor, choose the cheapest/fastest model class that clears the floor for a given query type. This is the synthesis lesson — every prior lesson produced a column of measured data against real Florida Building Code and Naples amendment questions; this lesson turns those columns into a decision function instead of a static report. The discipline underlying every technique surveyed across this series — SFT, RLHF, DPO, instruction tuning, small-model training — is the same: capability claims are established through empirical evaluation against a baseline, not asserted [1]. Your dashboard should embody that same discipline: no model-class recommendation without a measured basis specific to your evaluation set.

## 2. Why it matters for routing decisions

This lesson makes the project operationally useful rather than just a benchmark report. A decision engine that takes `{query_category, latency_budget_ms, cost_ceiling_usd, accuracy_floor}` and returns a ranked recommendation — including whether grounding is mandatory for that category — is the difference between "here are some numbers" and "here is a routing policy a permitting-software vendor could actually deploy."

## 3. Build increment

Add `engine/router.py`:

```python
import pandas as pd

def recommend_model_class(
    results_df: pd.DataFrame,
    query_category: str,   # "definitional" | "numeric" | "state_amendment" | "jurisdiction_amendment" | "diagram"
    latency_budget_ms: float,
    cost_ceiling_usd: float,
    accuracy_floor: float,
) -> dict:
    candidates = results_df[
        (results_df["query_category"] == query_category)
        & (results_df["p95_latency_ms"] <= latency_budget_ms)
        & (results_df["cost_per_1k_usd"] <= cost_ceiling_usd)
        & (results_df["citation_accuracy"] >= accuracy_floor)
    ]
    if candidates.empty:
        return {"recommendation": None, "reason": "no model class clears all constraints"}
    best = candidates.sort_values("cost_per_1k_usd").iloc[0]
    return {
        "recommendation": best["model_class"],
        "model_name": best["model_name"],
        "grounding_required": bool(best["grounded"]),
        "measured_citation_accuracy": best["citation_accuracy"],
        "measured_p95_latency_ms": best["p95_latency_ms"],
        "measured_cost_per_1k_usd": best["cost_per_1k_usd"],
    }
```

Build the dashboard front end (Streamlit is fastest):

```python
import streamlit as st

st.title("Florida Building Code — Model Routing Dashboard")
query_category = st.selectbox(
    "Query type", ["definitional", "numeric", "state_amendment", "jurisdiction_amendment", "diagram"]
)
latency_budget = st.slider("Latency budget (ms)", 100, 10000, 2000)
cost_ceiling = st.slider("Cost ceiling ($/1K requests)", 0.0, 50.0, 5.0)
accuracy_floor = st.slider("Citation accuracy floor", 0.0, 1.0, 0.9)

result = recommend_model_class(
    results_df, query_category, latency_budget, cost_ceiling, accuracy_floor
)
st.json(result)
st.bar_chart(results_df.set_index("model_class")[["p95_latency_ms", "cost_per_1k_usd"]])
```

## 4. Checkpoint — series complete

> **Implemented as a routing-behavior comparison, not a router product.** Rather than ship yet another model router into a crowded market (RouteLLM, NotDiamond, Martian, …), Lesson 7 **dry-runs** several routers over the eval set — recording *which model each picks*, not the answer — and plots the selections on **model-strength × grounding**. See [`LESSON7_PLAN.md`](../LESSON7_PLAN.md), [`routers/`](../routers/), [`run_router_dryrun.py`](../run_router_dryrun.py), and [`analysis/router_comparison.py`](../analysis/router_comparison.py). RouteLLM's ML stack won't install on Windows, so the routers run in a Linux container ([`docker/`](../docker/), [`docker-compose.yml`](../docker-compose.yml)) — which also makes the comparison one-command reproducible.
>
> **Measured result.** Off the shelf, **RouteLLM** (89% cheap / 11% strong) and **NotDiamond** (42% strong / 58% cheap) both route **cold** — 100% ungrounded — because grounding isn't a lever they expose. But grounding and model-tier are **orthogonal layers**: retrieve the passage yourself and hand the *grounded* prompt to NotDiamond and it climbs into the grounded band — except its cost model reads the longer prompt as *harder* and picks the **expensive** tier (strong 43/44, ~61%). **Training fixes that.** On our 44 grounded prompts the cheap model (Haiku) was never worse than strong (equal on 43, better on 1), so a NotDiamond custom router trained on those citation scores routes **cheap + grounded on all 44 (~77%)** — matching, even edging, the hand-built custom router (~71%, which routes a third of questions to a free *local* phi3). So the honest conclusion isn't "our router wins": it's that **the grounding decision belongs in front of routing**, and once you've made it a *trained* off-the-shelf router recovers cheap+grounded on its own. The custom router's only durable edge is **cost** — it can route to a free local model, which NotDiamond's hosted catalog cannot. Full synthesis + parameter provenance in [`FINDINGS.md`](../FINDINGS.md) § Lesson 7; the training CSV/code is walked through below.

### NotDiamond custom-router training — the CSV and the code

Because "just customize NotDiamond instead of building a router" is the natural objection, Lesson 7 actually runs it ([`run_notdiamond_training.py`](../run_notdiamond_training.py), in the Docker image). The pipeline:

1. **Build the training set** — run both candidate models on the 44 *grounded* prompts and score each answer with our citation-match metric (1.0 correct / 0.0 otherwise).
2. **Write NotDiamond's required CSV** (verified against the `notdiamond` 1.7.0 SDK docstring):
   ```
   prompt,anthropic/claude-sonnet-4-5/score,anthropic/claude-sonnet-4-5/response,anthropic/claude-haiku-4-5-20251001/score,anthropic/claude-haiku-4-5-20251001/response
   ```
   One `prompt` column (name passed as `prompt_column`), and for **each** candidate two columns named exactly `{provider}/{model}/score` and `{provider}/{model}/response`. Minimum 25 rows (we have 44). Write **only** those columns — extra columns can fail validation.
3. **`train_custom_router`** uploads the CSV and trains **server-side on NotDiamond's cloud** (`maximize=True`, since higher citation-score is better), returning a `preference_id`.
4. **Re-route** the 44 grounded prompts with `select_model(..., preference_id=..., tradeoff="cost")` so a router that learned "Haiku is as good as Sonnet when grounded" will pick the cheap one.

**Two gotchas worth knowing.** (a) *Candidate models must be callable by your key **and** in NotDiamond's catalog.* Our Anthropic key can't call the `claude-3.x` models NotDiamond lists, and `opus-4-7/4-8` aren't in its catalog — the overlap is the `4-5/4-6` line, so we use `claude-sonnet-4-5` (strong) and `claude-haiku-4-5-20251001` (cheap, the benchmark's own Haiku tier). (b) *Training is asynchronous* — `select_model(preference_id=…)` silently falls back to the **default** router until training finishes. Our first re-route (4 s later) reported strong 43/44; re-routing once training settled gave the real answer, cheap **44/44**. Always re-route after the job completes ([`--reroute-only`](../run_notdiamond_training.py)).

On NotDiamond's Pay-as-you-go plan this costs **$0** (custom-router training is included; 3 free) plus ~$0.30 of Anthropic inference to build the set. Caveat: training uploads the grounded prompts — which contain the short FBC excerpts — to NotDiamond's servers, a step outside the "excerpts stay local" boundary the rest of the project keeps.

At this point you have:

- Three (or four, with multimodal) working model clients measured against real FBC and Naples code questions
- A documented baseline hallucination rate per model class, cold
- A grounded-vs-ungrounded comparison quantifying what RAG buys you per model class
- A fine-tuning breakeven calculator using your own measured costs
- A constraint-based recommendation engine with an interactive dashboard
- A committed evaluation dataset and results (CSV/Parquet) — the actual evidence

## 5. Publish

1. Push the repo (MIT license), including `data/fbc_eval_questions.csv` and `results/*.jsonl` — reproducibility is the credibility mechanism, and a domain-expert reader can independently verify your gold answers against the published code.
2. Include a clear, prominent disclaimer: this is a research/benchmarking tool, not a substitute for consulting the Naples/Collier County building department or a licensed design professional for actual compliance decisions.
3. Write the accompanying post leading with the measured finding, e.g., "An ungrounded foundation model invented a Florida Building Code section number in X% of jurisdiction-specific questions — grounding cut that to Y%." Specific, measured, falsifiable claims outperform generic "LLMs are good at code lookup now" takes.
4. Explicitly document what did NOT work — categories where no model class cleared the accuracy floor even with grounding. This is what separates credible technical writing grounded in a real system from marketing content.

## References

1. Wang, Z. et al. "A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More." arXiv:2407.16216. https://arxiv.org/abs/2407.16216
