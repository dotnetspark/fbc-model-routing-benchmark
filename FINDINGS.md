# Findings — Can an LLM correctly cite the Florida Building Code?

**A measured comparison across three model tiers, cold and with retrieval grounding, on Florida Building Code and Naples/Collier County local code.**

> **TL;DR.** From memory alone, none of the three model tiers is usable for code-compliance citation — the best cites the correct section on about a quarter of questions, and the cheapest on under a tenth. They fail *differently* (the flagship abstains, the cheap hosted model fabricates, the tiny local model simply doesn't know). Injecting the correct code passage as context changes everything: it lifts every tier by 33–48 points and, crucially, **rescues the cheap local model** — a ~3.8B model running on a laptop goes from 7.9% to 52.5% correct, beating the *cold* flagship at near-zero marginal cost. The production lesson: **cheap local model + good retrieval beats an expensive model alone**, and the model tier matters far less than whether you ground it.

This document is the synthesis. Per-lesson detail (concept, method, full reasoning) lives in each lesson's `## 4. Checkpoint` in [`lessons/`](lessons/); every number here regenerates from [`analysis/benchmark_report.ipynb`](analysis/benchmark_report.ipynb) over the raw results in [`results/`](results/).

## The experiment

- **Task.** 45 real questions about the 2023 8th-Edition Florida Building Code and Naples/Collier County local code, each with a gold section citation verified against the published text. 22 are local-jurisdiction questions (the narrow, low-traffic material), the rest statewide FBC. See [`data/fbc_eval_questions.csv`](data/fbc_eval_questions.csv).
- **Metric.** Deterministic **section-citation match** (does the model cite the section the published code assigns?), split three ways: **correct**, **wrong/invented** (hallucination), **no citation** (abstention). A regex extractor, not an LLM judge, so the signal is independent of any model.
- **Tiers.** *Foundation* = Claude Opus 4.8 (flagship). *Instruction-tuned* = Claude Haiku 4.5 under a required-`section` JSON schema. *SLM* = phi3:mini (~3.8B) on local Ollama.
- **Conditions.** *Cold* (parametric memory only, 45 questions × 3 repeats) vs. *grounded* (the correct code passage injected as context, 44 questions; q017 is a FEMA date with no code section).

## Headline result

Citation outcomes, re-scored against the verified gold set:

| tier | cold — correct / wrong / abstain | grounded — correct / wrong / abstain | Δ correct |
|---|---|---|---|
| Opus 4.8 (foundation) | 25.8% / 32.6% / 41.7% | **59.1%** / 11.4% / 29.5% | **+33** |
| Haiku 4.5 (instruction-tuned) | 29.5% / 69.7% / 0.8% | **77.3%** / 22.7% / 0.0% | **+48** |
| phi3:mini (SLM, local) | 7.9% / 39.4% / 52.8% | **52.5%** / 12.5% / 35.0% | **+45** |

## The findings

### 1. Cold, no tier is usable — and each fails in a distinct way
Parametric knowledge of a frequently-amended, jurisdiction-specific code is exactly what these models are worst at. But the *shape* of failure differs by tier, and the difference is the alignment story made visible:
- **The flagship abstains.** Opus says "I don't have Section 202 memorized with enough precision" 42% of the time — calibrated honesty about the absence of knowledge (RLHF/DPO added calibration, not code knowledge).
- **The forced-schema model fabricates.** Haiku, required by its JSON schema to always name a section, can never say "I don't know," so uncertainty becomes confident invention — 70% wrong citations.
- **The small model simply doesn't know.** phi3 has almost no FBC exposure and none of the local amendments; it's honest (abstains more than it fabricates) but correct only 7.9% of the time.

Worst everywhere: the **local-jurisdiction** questions (Naples/Collier ordinances), the narrowest knowledge — precisely where retrieval will matter most.

### 2. The structured-output trap — and its reversal
The JSON schema that makes a permitting integration possible is a **liability cold and an asset grounded**. Cold, forcing a `section` field turned Haiku into the worst tier (it *must* cite, so it invents). Grounded, the exact same forced-citation mechanism makes Haiku the *best* tier (77.3%) — with the right passage present, being made to cite means being made to cite correctly. A schema guarantees a *parseable* answer, never a *correct* one; what flips its value is whether the context is there.

### 3. Grounding rescues the cheap local model — the economic headline
Retrieval is the highest-leverage intervention in the series. The SLM gains **+45 points (6.6×)**, and **grounded phi3 (52.5%) beats *cold* Opus (25.8%) by roughly 2×** — at near-zero marginal cost and fully offline. Cold, the SLM trailed the flagship by ~18 points; grounded, the gap shrinks to ~7. This is the production architecture the whole series argues toward: **a cheap local model with good retrieval, not an expensive model alone.**

### 4. Grounding is necessary, not sufficient
Even with the correct passage in front of it, a model sometimes still cites a section that isn't there. **Grounding-compliance** (cited a section actually present in the passage) tops out at 84% (Haiku) and falls to 62% (SLM). Treat grounding as risk-reduction, not a guarantee — and note the flagship's caution now *costs* it: grounded Opus still abstains ~30%, leaving accuracy on the table that the forced-citation Haiku captures.

### 5. Meta-finding: even our own gold set needed verification
Building the grounding corpus against the *published* text caught errors in the gold answers we had written from memory: three statewide FBC section numbers were wrong (§1020.2→1020.3 corridor width, §1003.2→1003.3.1 headroom, §1804.4→1804.5 flood-hazard fill), plus a local Pre-FIRM date. This is the project's own thesis turned on its authors — *do not trust memory for code citations, verify against primary text* — and it's a caution for anyone building an eval set.

## Exhibit A — what cold failure looks like
Full list in [`results/hallucinated_citations.csv`](results/hallucinated_citations.csv). Three archetypes recur, not equally dangerous:
- **Base-code substitution** — answering the *Florida* question from the generic ICC model code seen more in pretraining (a model literally opens "According to the International Building Code…" on an FBC question).
- **Plausible near-misses** — the right neighborhood, wrong section (`62-81` for `62-80`, `105.2.2` for `105.2.4`) — the worst kind for a compliance tool because they pass a smell test.
- **Outright fabrication** — inventing an ordinance that doesn't exist ("Collier County Ordinance 2021-03," complete with a fake adoption history), a 300-ft coastal buffer (actual: 100 ft), a 30-ft setback (actual: 75 ft).

## Caveats
- **Small category counts.** `state_amendment` (n=3) and `definitional` (n=4) are directional, not headline. The 22 jurisdiction and 16 numeric questions carry the weight.
- **Repeats aren't independent.** Three calls to frozen weights are correlated; effective n ≈ questions, not questions × repeats. Grounded runs are a single pass over 44 questions.
- **Manual retrieval.** Grounding uses the *correct* hand-verified passage per question, so this measures the *ceiling* of grounding, not a full RAG pipeline (retrieval error would lower it). That is the intended experiment — does the right text change the answer — not an end-to-end RAG benchmark.
- **The SLM is CPU-bound.** phi3 ran at ~5 tokens/sec, ~68 s/request. The near-zero *marginal* cost is real; the latency is the true cost without a GPU.
- **FBC text is ICC-copyrighted.** Only short per-question excerpts ship in [`data/fbc_eval_context.csv`](data/fbc_eval_context.csv); reproduce the full passages from UpCodes (see the notebook / Lesson 6).

## Reproduce
```bash
pip install -r requirements.txt        # or: anthropic google-genai python-dotenv pandas matplotlib jupyter pymupdf pydantic
python run_benchmark.py --model-class foundation                 # cold
python run_benchmark.py --model-class foundation --grounded      # grounded (needs data/fbc_eval_context.csv)
jupyter nbconvert --to notebook --execute --inplace analysis/benchmark_report.ipynb
```
