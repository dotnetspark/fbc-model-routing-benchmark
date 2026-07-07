# Lesson 6 — RAG-Integrated Models

## 1. Concept

A "RAG-integrated model" is not a distinct architecture — it's any of the model classes from Lessons 1–4 operating inside a pipeline that retrieves relevant code text before generation and constrains the model to ground its answer in that text. The model is unchanged; the inference-time contract changes: answer only from supplied context, and cite which passage supports the claim.

This is the single highest-leverage lesson in the series for a code-compliance assistant, because it directly targets the failure mode measured in every prior lesson: invented or wrong section citations. A full advanced-RAG build (query rewriting, hybrid search, reranking across the entire FBC corpus) is a follow-on project — this lesson only tests whether *any* retrieved context changes each model class's citation accuracy, using manually curated context passages.

## 2. Why it matters for routing decisions

If grounding closes most of the gap between the SLM and the foundation model on citation accuracy, that's the strongest possible argument for a cost-efficient production architecture: cheap model + good retrieval, rather than expensive model alone. If it doesn't close the gap — if the SLM still garbles cross-references and exceptions even with the right text in front of it — that tells you retrieval alone isn't sufficient and you need either a stronger model or the fine-tuning path from Lesson 5.

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

For each of your 25–40 evaluation questions, attach the actual FBC or Naples amendment text (200–500 words) containing the answer, sourced directly from the published code — not paraphrased, not from model memory. Run the grounded variant across all three model classes. Add a grounding-compliance check: does the cited section number actually match a section number present in the supplied context (not just a plausible-looking one)?

## 4. Checkpoint

You now have paired (ungrounded, grounded) results per model class across your full evaluation set. Compute the citation-accuracy delta from grounding for each class. This is your headline chart: expect the SLM to show the largest relative improvement, closing most of the gap to the foundation model on straightforward numeric lookups, while jurisdiction-amendment questions with dense cross-references may still favor the stronger model even when grounded.

**Next:** Lesson 7 closes the loop with a decision engine that recommends a model class and grounding requirement, given the query type.
