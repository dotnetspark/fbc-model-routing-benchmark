# The model tier matters far less than whether you ground it — I measured it on building code

Everyone building with LLMs eventually asks the same two questions: *how big a model do I actually need?* and *do I need retrieval?* I got tired of answering them with intuition, so I ran a small, fully reproducible benchmark on a task where the answer is checkable — **can a model correctly cite the section of the Florida Building Code that governs a question?** — and the results reorder how I'd architect these systems.

Full data, code, and results (MIT-licensed): https://github.com/dotnetspark/fbc-model-routing-benchmark

## The test

45 real questions about the 2023 Florida Building Code and Naples/Collier County local amendments, each with a gold section citation **verified against the published text**. I scored answers with a deterministic metric — did the model cite the section the code actually assigns? — split into **correct**, **wrong (hallucination)**, and **abstained (said "I don't know")**. Three model tiers: a flagship (Claude Opus 4.8), a cheap hosted model under a strict JSON schema (Claude Haiku 4.5), and a ~3.8B model running locally (phi3:mini). Two conditions: from memory, and with the correct code passage retrieved into context.

## Three findings a team should internalize

**1. From memory, none of them is usable — and they fail in dangerously different ways.** The flagship *abstains* (honest: "I don't have that section memorized"). The cheap schema-bound model, forced by its JSON contract to always name a section, *fabricates* — 70% wrong citations, because it can never say "I don't know." The small model simply doesn't know. The lesson: a structured-output contract guarantees a *parseable* answer, not a *correct* one, and it can actively suppress your safest failure mode. In a compliance setting, a confident wrong citation is far more expensive than an honest abstention.

**2. Retrieval is the highest-leverage change — and it rescues the *cheap, local* model.** Injecting the correct passage lifted every tier by 33–48 points. The headline: **a ~3.8B model running on a laptop, grounded, scored 52.5% — beating the flagship's 25.8% from memory, at roughly zero marginal cost and fully offline.** The production takeaway isn't "buy the biggest model." It's **cheap local model + good retrieval beats an expensive model alone.** The tier you pick matters far less than whether you ground it.

**3. You probably don't need to build a custom router.** The tempting next step is a "model router" that sends easy questions to the cheap model and hard ones to the flagship. But every off-the-shelf router I tested optimizes the wrong axis — *which model* — while the decision that actually moves the needle is *whether to ground*. Those are orthogonal: grounding is a retrieval step that lives in front of model selection, not inside it. When I gave a router the grounded prompt and then **trained** it on my own evaluation scores, it learned to route to the cheap model *and* stay grounded — matching a hand-built, domain-specific router without the bespoke engineering. Compose retrieval with a trained off-the-shelf router; save the custom build.

## The part I'm most glad I did

Building the retrieval corpus against the *published* code caught errors in my *own* gold answers — section numbers and a date I'd written from memory that were subtly wrong. The whole project's thesis ("don't trust memory for code citations — verify against the primary text") turned on its author. If you're building an evaluation set, verify every "correct" answer against the source. It's tedious and it's the difference between a credible benchmark and a confident-sounding wrong one.

## What this means if you're shipping

- **Default to retrieval before you default to a bigger model.** It's cheaper and higher-leverage.
- **A cheap local model + solid retrieval is a real production architecture**, not a toy — especially where cost, latency, privacy, or offline operation matter.
- **Be careful with forced structured output on knowledge tasks** — it can convert "I don't know" into confident fabrication.
- **Put the grounding decision first**; treat model-tier routing as a secondary optimization you can hand to a trained off-the-shelf tool.

This is a research benchmark, not compliance advice — nobody should pull a permit based on it, and it doesn't replace a licensed professional or the local building department. But as a measured answer to "how big a model, and do I need RAG," it was clarifying: **ground first, and the model tier gets a lot less interesting.**

*Everything is reproducible from the repo — I'd genuinely welcome being shown where I got it wrong.*
