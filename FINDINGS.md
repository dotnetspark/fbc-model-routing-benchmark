# Findings — Can an LLM correctly cite the Florida Building Code?

**A measured comparison across three model tiers, cold and with retrieval grounding, on Florida Building Code and Naples/Collier County local code.**

> **TL;DR.** From memory alone, none of the three model tiers is usable for code-compliance citation — the best cites the correct section on under a third of questions, and the cheapest local model on under a tenth. They fail *differently* (the flagship abstains, the cheap hosted model fabricates, the tiny local model simply doesn't know). Injecting the correct code passage as context changes everything: it lifts every tier by 33–46 points and, crucially, **rescues the cheap local model** — a ~3.8B model running on a laptop goes from 7.9% to 52.5% correct, beating the *cold* flagship at near-zero marginal cost. The production lesson: **cheap local model + good retrieval beats an expensive model alone**, and the model tier matters far less than whether you ground it.

This document is the synthesis. Per-lesson detail (concept, method, full reasoning) lives in each lesson's `## 4. Checkpoint` in [`lessons/`](lessons/); every number here regenerates from [`analysis/benchmark_report.ipynb`](analysis/benchmark_report.ipynb) over the raw results in [`results/`](results/).

## The experiment

- **Task.** 45 real questions about the 2023 8th-Edition Florida Building Code and Naples/Collier County local code, each with a gold section citation verified against the published text. 22 are local-jurisdiction questions (the narrow, low-traffic material), the rest statewide FBC. See [`data/fbc_eval_questions.csv`](data/fbc_eval_questions.csv).
- **Metric.** Deterministic **section-citation match** (does the model cite the section the published code assigns?), split three ways: **correct**, **wrong/invented** (hallucination), **no citation** (abstention). A regex extractor, not an LLM judge, so the signal is independent of any model.
- **Tiers.** *Foundation* = Claude Opus 4.8 (flagship). *Instruction-tuned* = Claude Haiku 4.5 under a required-`section` JSON schema. *SLM* = phi3:mini (~3.8B) on local Ollama.
- **Conditions.** *Cold* (parametric memory only, 45 questions × 3 repeats) vs. *grounded* (the correct code passage injected as context, 44 questions; q017 is a FEMA date with no code section).

## Headline result

Citation outcomes, scored against the verified gold set. Denominators are *scoreable answers*: q017 (a date with no code section) and failed API requests are excluded — cold n = 132/132/127, grounded n = 44/44/40. These are exactly the numbers the notebook prints.

| tier | cold — correct / wrong / abstain | grounded — correct / wrong / abstain | Δ correct |
|---|---|---|---|
| Opus 4.8 (foundation) | 26.5% / 31.8% / 41.7% | **59.1%** / 11.4% / 29.5% | **+32.6** |
| Haiku 4.5 (instruction-tuned) | 31.8% / 67.4% / 0.8% | **77.3%** / 22.7% / 0.0% | **+45.5** |
| phi3:mini (SLM, local) | 7.9% / 39.4% / 52.8% | **52.5%** / 12.5% / 35.0% | **+44.6** |

## The findings

### 1. Cold, no tier is usable — and each fails in a distinct way
Parametric knowledge of a frequently-amended, jurisdiction-specific code is exactly what these models are worst at. But the *shape* of failure differs by tier, and the difference is the alignment story made visible:
- **The flagship abstains.** Opus says "I don't have Section 202 memorized with enough precision" 42% of the time — calibrated honesty about the absence of knowledge (RLHF/DPO added calibration, not code knowledge).
- **The forced-schema model fabricates.** Haiku, required by its JSON schema to always name a section, can never say "I don't know," so uncertainty becomes confident invention — 67% wrong citations.
- **The small model simply doesn't know.** phi3 has almost no FBC exposure and none of the local amendments; it's honest (abstains more than it fabricates) but correct only 7.9% of the time.

Worst everywhere: the **local-jurisdiction** questions (Naples/Collier ordinances), the narrowest knowledge — precisely where retrieval will matter most.

### 2. The structured-output trap — and its reversal
The JSON schema that makes a permitting integration possible is a **liability cold and an asset grounded**. Cold, forcing a `section` field turned Haiku into the worst tier (it *must* cite, so it invents). Grounded, the exact same forced-citation mechanism makes Haiku the *best* tier (77.3%) — with the right passage present, being made to cite means being made to cite correctly. A schema guarantees a *parseable* answer, never a *correct* one; what flips its value is whether the context is there.

### 3. Grounding rescues the cheap local model — the economic headline
Retrieval is the highest-leverage intervention in the series. The SLM gains **+45 points (6.6×)**, and **grounded phi3 (52.5%) beats *cold* Opus (26.5%) by roughly 2×** — at near-zero marginal cost and fully offline. Cold, the SLM trailed the flagship by ~19 points; grounded, the gap shrinks to ~7. This is the production architecture the whole series argues toward: **a cheap local model with good retrieval, not an expensive model alone.**

### 4. Grounding is necessary, not sufficient
Even with the correct passage in front of it, a model sometimes still cites a section that isn't there. **Grounding-compliance** (cited a section actually present in the passage) tops out at 84% (Haiku) and falls to 62% (SLM). Treat grounding as risk-reduction, not a guarantee — and note the flagship's caution now *costs* it: grounded Opus still abstains ~30%, leaving accuracy on the table that the forced-citation Haiku captures.

### 5. Meta-finding: even our own gold set needed verification
Building the grounding corpus against the *published* text caught errors in the gold answers we had written from memory: three statewide FBC section numbers were wrong (§1020.2→1020.3 corridor width, §1003.2→1003.3.1 headroom, §1804.4→1804.5 flood-hazard fill), plus a local Pre-FIRM date. This is the project's own thesis turned on its authors — *do not trust memory for code citations, verify against primary text* — and it's a caution for anyone building an eval set.

## Exhibit A — what cold failure looks like
Full list in [`results/hallucinated_citations.csv`](results/hallucinated_citations.csv). These are drawn from the cold runs across tiers — including a **second, free-tier foundation model (Gemini 3 Flash)** we ran alongside Opus; the vivid fabrications below are *its* output (`results/foundation_gemini3flash_raw.jsonl`), not the headline Opus run's, which fails more by abstaining than by inventing. Three archetypes recur, not equally dangerous:
- **Base-code substitution** — answering the *Florida* question from the generic ICC model code seen more in pretraining (Gemini 3 Flash literally opens "According to the International Building Code…" on an FBC question — and the local SLM does the same).
- **Plausible near-misses** — the right neighborhood, wrong section (`62-81` for `62-80`, `105.2.2` for `105.2.4`) — the worst kind for a compliance tool because they pass a smell test.
- **Outright fabrication** — Gemini 3 Flash inventing an ordinance that doesn't exist ("Collier County Ordinance 2021-03," complete with a fake adoption history), a 300-ft coastal buffer (actual: 100 ft), a 30-ft setback (actual: 75 ft).

## Lesson 5 — routing behavior: grounding is a separate axis, in front of model choice

The obvious closer would be "build a model router." But the market is crowded (RouteLLM, NotDiamond, Martian, Unify, OpenRouter), and they all route on one axis — *prompt → model tier*. So instead of shipping another router, we **dry-ran** several routers over the 45 questions — recording *which model each picks, and whether it grounds*, not the answer — and plotted the selections on two axes: **model strength** and **grounded?** ([`results/router_selection_gravity.png`](results/router_selection_gravity.png), from [`analysis/router_comparison.py`](analysis/router_comparison.py); routers in [`routers/`](routers/), driver [`run_router_dryrun.py`](run_router_dryrun.py)).

**Off the shelf, every router routes cold — 100% ungrounded, ~30% expected accuracy:**
- **RouteLLM** (local `bert` win-rate classifier): 89% cheap / 11% strong. *(Its ML stack won't install on Windows, so it runs in a Linux container — `docker compose run --rm routers` — which also makes the comparison reproducible.)*
- **NotDiamond** (`model_router.select_model`): 42% strong / 58% cheap.
- **A transparent difficulty heuristic** (*illustrative*, 38 readable lines): 44/56 — ungrounded *by construction* (`grounded=False` hardcoded); it just makes the mechanism legible.

They aren't blind to grounding because they're bad — they're blind because **grounding and model-tier are orthogonal**. Grounding is a *pipeline* decision (retrieve a passage, inject it) that lives outside the "pick a model for these messages" abstraction these routers operate on.

**So we tested the sharpest objection directly — can you *leverage* NotDiamond instead of hand-building a router?** Two steps (both real, in [`routers/notdiamond_grounded.py`](routers/notdiamond_grounded.py) and [`run_notdiamond_training.py`](run_notdiamond_training.py)):

1. **Ground the prompt, keep the default router.** Hand NotDiamond the *grounded* prompt (passage injected). It climbs into the grounded band — but its cost model reads the longer prompt as *harder* and picks the **expensive** tier: **strong 43/44, ~61%**. Composition alone grounds you, but routes you to the costly model.
2. **Train a custom NotDiamond router** on those grounded prompts (its free `train_custom_router`). On our 44 prompts the cheap model (Haiku) was **never worse** than strong — equal on 43, better on 1 — so the trained router, with `tradeoff="cost"`, routes **cheap + grounded on all 44: ~77%**.

**The hand-built custom router** — a readable lookup table encoding the Lessons 1–4 findings — also sits cheap + grounded, at **~71%** (lower *only* because it routes a third of questions to a free *local* phi3 at ~70%, where the trained NotDiamond uses paid Haiku at ~77%).

The quadrant "gravity" map makes the arc visible: cold, the shaded GROUNDED half is empty for every off-the-shelf router; supply grounding and default NotDiamond enters it at **strong** (blue); **train it and it lands in exactly the same cheap+GROUNDED quadrant (green) as the custom router.**

**The honest conclusion — and a correction to an earlier draft of this doc.** It is *not* true that off-the-shelf routers "can't express grounding" — our own experiment disproves it. The accurate claim is about **layering**: grounding is a separate axis that sits *in front of* model-tier routing. Out of the box a router optimizes the *secondary* lever (which model) and leaves the *primary* one (whether to ground) to you; supply it, and a *trained* router even co-optimizes the tier for cost. So you do **not** need to hand-build a router — **retrieval + a trained off-the-shelf router matches (here, beats) the bespoke one on accuracy.** The bespoke router's only durable edge is **cost**: it can route to a *free local* model, which NotDiamond's hosted catalog cannot. Method, parameter provenance, and honesty guardrails below; the training pipeline is walked through in [Lesson 5](lessons/05-when-to-use-each-model-type.md).

### Design & parameter provenance (Lesson 5)

For anyone presenting this: the method is a **dry-run selection study** — run each router's *choice* over the 45 questions (which model, and would it ground) *without calling the model*, so it isolates the routing **policy** from answer quality at near-zero cost. Then plot strength × grounding. (The one exception is the NotDiamond *training* experiment, which calls the two candidate models once each to build the labeled set — ~$0.30.) Every parameter, and exactly what it rests on:

| Parameter | Value | Based on | Honest status |
|---|---|---|---|
| custom router's grounded-accuracy table | measured rates | Lessons 1–4 results | data-derived — but **circular for scoring**, so we don't lead with the score |
| custom `FLOOR` (min usable accuracy) | 0.50 | a product judgment call | stated as a choice, not derived |
| RouteLLM router | `bert` | local/offline (the `mf` router calls the OpenAI *embeddings* API per prompt) | real model output |
| RouteLLM strong/cheap threshold | win-rate ≥ 0.5 | neutral default | arbitrary — doesn't change the finding (no threshold adds a grounding axis) |
| NotDiamond **routing** candidates (cold) | gpt-4o / claude-3-7-sonnet vs gpt-4o-mini / claude-3-haiku | probed against NotDiamond 1.7.0's catalog (many strings 400) | verified accepted |
| NotDiamond tradeoff | `"cost"` | cost-aware routing | a knob; drives the trained router toward cheap when quality ties |
| NotDiamond **training** candidates | `claude-sonnet-4-5` / `claude-haiku-4-5-20251001` | must be callable by our key **and** in ND's catalog (the 4-5/4-6 overlap) | cheap candidate **is** the benchmark's Haiku → trained cheap picks map exactly |
| NotDiamond training score | citation-match (1.0/0.0), `maximize=True` | our deterministic metric | binary + nearly non-discriminative (cheap ≤ strong on 43/44) — which is *why* the trained router picks cheap |
| trained-router re-route | re-run **after** training settles | async: `select_model(preference_id)` uses the *default* router until training finishes | 4-s first pass = strong 43/44; settled pass = cheap 44/44 |
| difficulty heuristic threshold / markers | 2.0, hand-picked words | domain intuition | **illustrative only**, not derived from data |
| strong→Opus, cheap→Haiku (accuracy overlay) | — | a stated mapping | assumption for the overlay; **exact** for custom's phi3 picks and the trained router's Haiku picks |

The move that makes it defensible: **separate the parameters that could be accused of "tuning to win" from the finding.** The empty *cold* grounded band doesn't depend on a threshold; and the training result (cheap 44/44) is *forced by the data* — the cheap model was never worse when grounded.

## Caveats
- **Small category counts.** `state_amendment` (n=3) and `definitional` (n=4) are directional, not headline. The 22 jurisdiction and 16 numeric questions carry the weight.
- **Repeats aren't independent.** Three calls to frozen weights are correlated; effective n ≈ questions, not questions × repeats. Grounded runs are a single pass over 44 questions.
- **Manual retrieval.** Grounding uses the *correct* hand-verified passage per question, so this measures the *ceiling* of grounding, not a full RAG pipeline (retrieval error would lower it). That is the intended experiment — does the right text change the answer — not an end-to-end RAG benchmark.
- **The SLM is CPU-bound.** phi3 ran at ~5 tokens/sec, ~68 s/request. The near-zero *marginal* cost is real; the latency is the true cost without a GPU.
- **FBC text is ICC-copyrighted.** Only short per-question excerpts ship in [`data/fbc_eval_context.csv`](data/fbc_eval_context.csv); reproduce the full passages from UpCodes (see the notebook / Lesson 4).

## Reproduce
```bash
pip install -r requirements.txt                                  # pinned; Python 3.11+
python run_benchmark.py --model-class foundation                 # cold (Opus; or foundation_gemini for $0)
python run_benchmark.py --model-class foundation --grounded      # grounded (needs data/fbc_eval_context.csv)
docker compose run --rm routers                                  # Lesson 5 router dry-run (Linux container)
jupyter nbconvert --to notebook --execute --inplace analysis/benchmark_report.ipynb
```
The full command list (all three tiers, grounded runs, NotDiamond training) is in the [README](README.md#reproducing-the-full-benchmark).
