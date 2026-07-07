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
    query_category: str,   # "definitional" | "numeric" | "jurisdiction_amendment" | "diagram"
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
    "Query type", ["definitional", "numeric", "jurisdiction_amendment", "diagram"]
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
