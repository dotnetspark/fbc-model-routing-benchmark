# Lesson 4 — Multimodal Models

## 1. Concept

Multimodal models extend the transformer architecture to accept image tokens alongside text, projecting non-text inputs into the same embedding space the language model reasons over. Preference-alignment techniques developed for text-only models transfer to this setting with modification — dense, segment-level preference optimization applied to multimodal LLMs has been shown to sharply reduce hallucination, with one method (RLHF-V) achieving a 34.8% reduction in hallucination rate using only 1.4k annotated preference samples, outperforming a concurrent method trained on roughly 7x more data [1]. This is directly relevant here: multimodal hallucination (describing a setback line or dimension that isn't actually on a site plan) is a distinct failure mode from text hallucination and needs its own detection strategy — it's the visual analog of the invented section-number problem from Lessons 1–3.

Small multimodal models are converging with the SLM trend — Microsoft's Phi-3 family extended to a dedicated vision variant, phi-3.5-Vision, alongside phi-3.5-mini and phi-3.5-MoE, specifically to add multimodal and long-context capability to the same small-model line [2], meaning the cost/capability tradeoff you've measured for text extends into diagram-reading tasks without requiring a separate evaluation framework.

## 2. Why it matters for routing decisions

Building code questions frequently reference diagrams: egress path illustrations, wind zone maps, flood elevation certificates, setback and lot-coverage diagrams. A text-only pipeline simply cannot answer "does this site plan meet the Naples front setback requirement?" Do not route every question through a multimodal model by default — it carries a real latency/cost premium. Route conditionally, only when an image is present in the payload.

## 3. Build increment

Add a routing predicate and a fourth client:

```python
def requires_multimodal(payload: dict) -> bool:
    return "image" in payload or "site_plan_image" in payload

async def call_multimodal_model(question: str, image_b64: str, question_id: str, client):
    start = time.perf_counter()
    response = await client.messages.create(
        model="your-multimodal-model",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": question},
            ],
        }],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    # ... construct InferenceResult identically to Lesson 1, tagged model_class="multimodal"
```

Extend the evaluation set with 5–10 image-bearing questions using public-domain or self-drawn examples: a simple setback diagram, a wind-borne-debris-region map excerpt, an egress path floor plan. Pair each with a ground-truth measurement (e.g., "the diagram shows a 25-foot front setback"). Add a hallucination check: does the model report a dimension or feature not actually present in the image?

## 4. Checkpoint

Your routing predicate correctly diverts image payloads, and the dashboard reports a fourth column with measured accuracy on diagram-reading tasks plus the latency/cost premium versus text-only calls. This is a genuinely useful finding for any AEC-adjacent tool builder — flag it clearly in your writeup.

**Next:** Lesson 5 shifts from model selection to a build-vs-fine-tune economic model.

## References

1. Yu, T. et al. "RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback." arXiv:2312.00849. https://arxiv.org/abs/2312.00849
2. Abdin, M. et al. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." arXiv:2404.14219. https://arxiv.org/abs/2404.14219
