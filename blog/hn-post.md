# Can an LLM cite the building code? I measured it — cold vs. grounded, three model tiers

*Suggested HN title: "Grounded phi3 (3.8B, on a laptop) beats cold Opus at citing building code"*

I kept seeing "LLMs are great at code lookup now" and wanted a number, not a vibe. So I built a small, reproducible benchmark: **can a model correctly cite the section of the Florida Building Code (and the Naples/Collier County local amendments) that governs a given question?** Citation is a nice task to measure because it's *checkable* — either the model names the section the published code assigns, or it doesn't.

Repo (MIT, all data + results committed): https://github.com/dotnetspark/fbc-model-routing-benchmark

## Setup

- **45 questions** against the 2023 8th-Edition FBC and Naples/Collier local code, each with a gold section citation **verified against the published text**. 22 are narrow local-jurisdiction questions (ordinances, the LDC) — the low-traffic material.
- **Metric: deterministic section-citation match** via regex, split three ways — **correct / wrong (hallucination) / no-citation (abstention)**. No LLM judge, so the signal doesn't depend on any model's opinion.
- **Three tiers:** Claude Opus 4.8 (flagship), Claude Haiku 4.5 under a required-`section` JSON schema (cheap hosted), and phi3:mini (~3.8B) on local Ollama (cheap, offline).
- **Two conditions:** cold (parametric memory only) and grounded (the correct code passage injected as context).

## Finding 1: cold, no tier is usable — and each fails *differently*

| tier | correct | wrong | abstained |
|---|---|---|---|
| Opus 4.8 | 25.8% | 32.6% | **41.7%** |
| Haiku 4.5 (forced schema) | 29.5% | **69.7%** | 0.8% |
| phi3:mini (local) | 7.9% | 39.4% | 52.8% |

Same poor correctness, three different failure shapes, and the shape is the whole story:

- **The flagship abstains.** Opus says "I don't have that section memorized with enough precision" 42% of the time — calibrated honesty.
- **The forced-schema model fabricates.** Haiku's JSON schema *requires* a `section`, so it can never say "I don't know" — uncertainty becomes confident invention (70% wrong).
- **The small model just doesn't know** (8% correct).

The dangerous failure for a compliance tool is a *confident wrong citation*, which is exactly what the cheap schema-bound model produces most. A schema buys you a parseable answer, never a correct one.

## Finding 2: grounding rescues the cheap *local* model (the economic headline)

Inject the correct passage and everything moves — +33 to +48 points per tier:

| tier | cold → grounded (correct) |
|---|---|
| Opus 4.8 | 25.8% → 59.1% |
| Haiku 4.5 | 29.5% → **77.3%** |
| phi3:mini | 7.9% → **52.5%** |

The reordering is the point: **grounded phi3 (52.5%, running on a laptop, ~free, offline) beats *cold* Opus (25.8%) by ~2×.** The production lesson isn't "use the big model" — it's **cheap local model + good retrieval beats an expensive model alone.** Model tier matters far less than whether you ground.

(The same forced-JSON schema that made Haiku the *worst* tier cold makes it the *best* grounded — with the right passage present, being made to cite means being made to cite correctly.)

## Finding 3: don't build a router — grounding is a separate axis

The obvious next move is a model router. But every off-the-shelf router (I dry-ran RouteLLM and NotDiamond — recording *which* model each picks, not the answer) routes on one axis, prompt → model tier, and lands **100% ungrounded** (~30% expected accuracy). Not because they're bad — because **grounding and model-tier are orthogonal.** Grounding is a pipeline step (retrieve, inject) that lives outside the "pick a model for these messages" abstraction.

So I tested the obvious objection directly — can you *leverage* an off-the-shelf router instead of hand-building one?

- **Ground the prompt, keep the default router:** NotDiamond climbs into the grounded band but its cost model reads the longer prompt as *harder* and picks the **expensive** tier (strong 43/44, ~61%).
- **Train a custom NotDiamond router** (its free `train_custom_router`) on the grounded prompts scored by citation-match: on 43/44 the cheap model was *never worse* than the strong one, so the trained router routes **cheap + grounded on all 44 (~77%)** — matching a hand-built, domain-specific router (~71%).

Conclusion: **put the grounding decision in front of routing, then a *trained* off-the-shelf router recovers cheap+grounded on its own.** You don't need to build a router. (The hand-built one's only durable edge: it can route to a free *local* model, which a hosted router's catalog can't.)

## The meta-finding: my own gold set had errors

Building the grounding corpus against the *published* text caught three wrong section numbers and one wrong date in gold answers I'd written from memory (e.g. §1020.2→1020.3, §1003.2→1003.3.1). The project's own thesis — *don't trust memory for code citations, verify against primary text* — turned on its author. If you build an eval set for a citation task, verify every gold against the primary source; I didn't, at first, and it showed.

## Caveats (the honest ones)

- **Small n.** 45 questions; some categories (state amendments n=3, definitions n=4) are directional only. The 22 jurisdiction + 16 numeric questions carry the weight.
- **This measures the *ceiling* of grounding.** I inject the *correct* hand-verified passage, so it isolates "does the right text change the answer" — not an end-to-end RAG pipeline (retrieval error would lower it).
- **The local model is CPU-bound** (~5 tok/s, ~68 s/request). The near-zero *marginal* cost is real; latency is the true cost without a GPU.
- **The trained-router result is n=44 with a binary score.** It works *because* cheap was never worse than strong when grounded — that's the mechanism, not a tuned outcome.

**Disclaimer:** this is a research/benchmarking tool, not a substitute for a licensed design professional or the Naples/Collier building department. Do not make compliance decisions from it.

Every number regenerates from the raw results in the repo. Happy to be told I measured something wrong.
