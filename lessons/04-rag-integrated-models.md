# Lesson 4 — RAG-Integrated Models

## 1. Concept

A "RAG-integrated model" is not a distinct architecture — it's any of the model classes from Lessons 1–3 operating inside a pipeline that retrieves relevant code text before generation and constrains the model to ground its answer in that text. The model is unchanged; the inference-time contract changes: answer only from supplied context, and cite which passage supports the claim.

This is the single highest-leverage lesson in the series for a code-compliance assistant, because it directly targets the failure mode measured in every prior lesson: invented or wrong section citations. A full advanced-RAG build (query rewriting, hybrid search, reranking across the entire FBC corpus) is a follow-on project — this lesson only tests whether *any* retrieved context changes each model class's citation accuracy, using manually curated context passages.

## 2. Why it matters for routing decisions

If grounding closes most of the gap between the SLM and the foundation model on citation accuracy, that's the strongest possible argument for a cost-efficient production architecture: cheap model + good retrieval, rather than expensive model alone. If it doesn't close the gap — if the SLM still garbles cross-references and exceptions even with the right text in front of it — that tells you retrieval alone isn't sufficient and you need either a stronger model or a fine-tuning pass — deliberately out of scope for this series (see the scope note in the README).

## 3. Build increment

Add a context-injection variant to your existing harness. No vector database is required yet — just the correct FBC/Naples passage, retrieved manually for each evaluation question:

```python
def build_grounded_prompt(question: str, context: str) -> str:
    return f"""Answer the Florida Building Code question using ONLY the context below.
If the answer is not in the context, say "Not found in context."
Always cite the section number exactly as it appears in the context.

Context:
{context}

Question: {question}
"""

async def run_grounded_variant(question_id: str, question: str, context: str, client_fn):
    grounded_prompt = build_grounded_prompt(question, context)
    return await client_fn(grounded_prompt, f"{question_id}_grounded")
```

For each of your 25+ evaluation questions, attach the actual FBC or Naples amendment text (200–500 words) containing the answer, sourced directly from the published code — not paraphrased, not from model memory. Run the grounded variant across all three model classes. Add a grounding-compliance check: does the cited section number actually match a section number present in the supplied context (not just a plausible-looking one)?

## 4. Checkpoint

You now have paired (ungrounded, grounded) results per model class. Compute the citation-accuracy delta from grounding for each class — this is the series' headline chart.

### Measured result (reference run)

Cold vs. grounded citation-match rate, same 44 groundable questions (q017 excluded — a FEMA date with no code section; failed API requests also excluded, so cold n = 132/132/127 answers and grounded n = 44/44/40). Grounding = the correct FBC/Collier passage injected as context, sourced from published text.

| tier | cold | grounded | delta | grounded compliance¹ |
|---|---|---|---|---|
| Opus 4.8 (foundation) | 26.5% | **59.1%** | +32.6 | 70.5% |
| Haiku 4.5 (instruction-tuned) | 31.8% | **77.3%** | **+45.5** | 84.1% |
| phi3:mini (SLM, local) | 7.9% | **52.5%** | +44.6 | 62.5% |

¹ grounding compliance = fraction of *all* answers that cite a section *actually present in the supplied passage*. Abstentions count as non-compliant, so this single number mixes two different failures — see the decomposition bullet below before quoting it.

**Grounding is the highest-leverage intervention in the series — it lifts every tier by 33–46 points.** Three findings drive the routing decisions in Lesson 5:

- **Retrieval rescues the cheap local model — this is the economic headline.** The SLM jumps from 7.9% to 52.5% (a 6.6× relative gain), with jurisdiction_amendment questions going 10.8% → 70.0%. **Grounded phi3 (52.5%) beats *cold* Opus (26.5%) by roughly 2×**, at near-zero marginal cost. Cold, the SLM trailed the flagship by ~19 points; grounded, the gap shrinks to ~7. "Cheap local model + good retrieval" is now a defensible production architecture — exactly the bet Lesson 2 §2 flagged.
- **The JSON schema flips from liability to asset.** In Lesson 2 the required-`section` field made Haiku the *worst* tier cold (67% hallucination — forbidding "I don't know" turned uncertainty into fabrication). Grounded, that same forced-citation mechanism makes Haiku the *best* tier (77.3%): with the right passage present, being forced to cite means being forced to cite *correctly*. Meanwhile the flagship, even grounded, still abstains ~30% ("Not found in context") — its calibration, an asset cold, now leaves accuracy on the table.
- **Grounding is necessary but not sufficient — and the failure-mode split survives it.** Decomposing non-compliance against the raw rows: grounded Opus never cited an absent section (0 of 31 citations; its non-compliance is entirely "Not found in context" abstention, 13/44), grounded Haiku — which cannot abstain — cited an absent section in 15.9% of answers (7/44), and grounded phi3 mostly abstained (14/40) with one invented citation (1 of 26). Residual hallucination is real but concentrated in the forced-schema tier: the *output contract*, not the model tier, selects which failure mode survives grounding. A decision engine should treat grounding as risk-reduction, not a guarantee, and weight it against query type (jurisdiction questions benefit most; the few dense cross-referenced ones still slip).

A methodological note worth publishing: assembling the grounding corpus **caught three errors in our own statewide gold sections** (q001 §1020.2→1020.3, q002 §1003.2→1003.3.1, q011 §1804.4→1804.5) that were written from memory and never checked against the published FBC — the same "trust the model's memory" failure this whole series measures, committed by the authors of the gold set. Verify gold against primary text, always.

**Next:** Lesson 5 closes the loop by comparing routing policies — off-the-shelf routers, a trained NotDiamond custom router, and a hand-built one — on the two axes that actually matter here: model strength × grounding.
