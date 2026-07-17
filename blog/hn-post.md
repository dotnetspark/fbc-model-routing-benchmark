# HN submission plan

**Submit as a Show HN *link* post to the repo** — https://github.com/dotnetspark/fbc-model-routing-benchmark — not as a text post. HN text posts don't render markdown (no tables, no headers), and the README now leads with the results table and charts, so the repo *is* the post. Then immediately add the first comment below.

**Title** (leads with the routing thesis, so the repo name, the README, and the post tell one story; the citation task appears as the method in the first comment. Also avoids inviting "you gave it the answer" as the top comment):

> Show HN: A model-routing benchmark — the routers optimize the wrong axis

Backup title (citation-first, if the routing frame feels too abstract on the day):

> Show HN: Measuring LLM citation accuracy on building code (3 tiers, cold vs. grounded)

**First comment** — paste the text below verbatim. It's written in HN-safe formatting: blank-line paragraphs, 4-space-indented blocks for aligned numbers, no tables or headers. It leads with method, preempts the strongest objection in the body instead of the footnotes, and ends with an invitation to attack the setup.

---

I wanted a measured answer to a routing question — which model tier should handle which query — and I suspected the honest answer would embarrass somebody, possibly me. So I built a routing benchmark on a task where correctness is checkable: can a model cite the section of the Florida Building Code (and the Naples/Collier County local amendments) that governs a question? Either it names the section the published code assigns, or it doesn't. A regex extractor does the scoring — no LLM judge, so the signal doesn't depend on any model's opinion. The punchline up front: the tier question turned out to be the wrong *first* question, and every off-the-shelf router I tested optimizes that secondary lever.

Setup: 45 questions, each with a gold citation verified against the published text (22 are narrow local-jurisdiction questions — the low-traffic material). Three tiers: Claude Opus 4.8 (flagship), Claude Haiku 4.5 forced through a JSON schema with a required "section" field (cheap hosted), and phi3:mini (~3.8B) on local Ollama. Two conditions: cold (parametric memory only) and grounded (the correct code passage injected as context). All data, raw results, and the analysis notebook are in the repo (MIT).

Cold — correct / wrong (hallucinated) / abstained:

    Opus 4.8:   26.5% / 31.8% / 41.7%
    Haiku 4.5:  31.8% / 67.4% /  0.8%
    phi3:mini:   7.9% / 39.4% / 52.8%

Similar (poor) correctness, three completely different failure shapes — and the shape is the story. The flagship abstains ("I don't have that section memorized with enough precision" — verbatim). The schema-bound model can never say "I don't know," so its uncertainty becomes confident invention: 67% wrong citations. The small model mostly just doesn't know. For a compliance tool, the dangerous failure is the confident wrong citation, which is exactly what the forced schema manufactures. A schema buys you a parseable answer, never a correct one.

Grounded, correct-citation rates become: Opus 59.1%, Haiku 77.3%, phi3 52.5%. The reordering is the point — grounded phi3, running on a laptop CPU, ~free and offline, beats cold Opus (26.5%) by about 2x. Cheap local model + good retrieval beat an expensive model alone, and the same JSON schema that made Haiku the worst tier cold makes it the best tier grounded.

The obvious objection: grounding injects the passage that contains the right section number, so this is close to a reading-comprehension task — it measures the *ceiling* of retrieval, not an end-to-end RAG pipeline. That's deliberate (the question was "does the right text change the answer"), but here's what surprised me: models still fail it. Even with the correct passage in front of them, 16–38% of cited sections aren't present in the supplied passage at all (grounding-compliance: Haiku 84%, Opus 70%, phi3 62%). Grounding is risk reduction, not a guarantee.

On routers: the tempting next step is a model router, but every off-the-shelf router I dry-ran (RouteLLM, NotDiamond — recording which model each picks, not the answers) routes 100% ungrounded, because grounding is a pipeline decision outside the "pick a model for these messages" abstraction. Ground the prompt yourself and NotDiamond's default router climbs into the grounded band but picks the expensive tier (it reads the longer prompt as harder). Train its free custom router on my own citation scores and it learns cheap + grounded on all 44 prompts (~77%) — matching a hand-built domain router (~71%). So: put the grounding decision in front of routing, and you probably don't need to build a router.

My favorite finding is the embarrassing one: building the grounding corpus against the published text caught three wrong section numbers and a wrong date in gold answers I'd written from memory. The project's own thesis — don't trust memory for code citations — turned on its author. If you build an eval set for a citation task, verify every gold answer against the primary source.

Caveats: n=45 (some categories are directional only); repeats of frozen weights aren't independent samples; the local model is CPU-bound (~68 s/request — the "free" is marginal cost, not latency); and this is a research benchmark, not compliance advice — nobody should pull a permit based on it.

Every number regenerates from the raw results in the repo (one notebook, committed outputs). Happy to be told I measured something wrong.
