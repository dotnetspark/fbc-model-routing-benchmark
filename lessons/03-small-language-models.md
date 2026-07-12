# Lesson 3 — Small Language Models (SLMs)

## 1. Concept

SLMs achieve disproportionate capability-per-parameter by prioritizing training-data quality over raw scale. Microsoft's Phi-3 family is the reference case: phi-3-mini, a 3.8 billion parameter model trained on 3.3 trillion tokens, has overall performance rivaling models such as Mixtral 8x7B and GPT-3.5 (69% on MMLU, 8.38 on MT-bench), despite being small enough to deploy on a phone [1]. This directly challenges naive scaling-law assumptions that hold data source fixed: the Phi-3 approach relies on a heavily filtered, curated training corpus plus synthetic data rather than raw scale, following the "textbooks are all you need" line of work [1]. Deployment implications are concrete: phi-3-mini can be quantized to 4 bits, occupying roughly 1.8GB of memory, and was demonstrated running fully offline on an iPhone 14 at over 12 tokens per second [1]. Parameter-scaled variants extend the family — phi-3-small (7B) and phi-3-medium (14B), trained for 4.8T tokens, reach 75% and 78% on MMLU respectively [1] — giving a capability dial within one family.

**Why this matters specifically for FBC lookups:** an SLM's pretraining corpus almost certainly under-represents the Florida Building Code relative to general web text, and it has essentially no exposure to Naples/Collier County local amendments — these are narrow, low-traffic municipal documents. Expect the SLM's cold (non-grounded) performance on jurisdiction-specific questions to be the weakest of all three model classes. That's the finding this lesson is designed to surface, not a flaw in your setup.

## 2. Why it matters for routing decisions

SLMs are the right default for high-volume, low-complexity tasks; latency-sensitive or on-device deployments (e.g., a field inspector's offline tablet app); and cost-sensitive pipelines. For a permitting-software vendor processing thousands of lookups a day, an SLM's near-zero marginal cost is compelling — *if* it can be paired with retrieval grounding to compensate for its narrow domain exposure. This lesson measures the gap that Lesson 6 will close.

## 3. Build increment

Add `clients/slm_client.py` using a local Ollama-served model (Phi-3-mini or Llama-3.2-3B):

```python
import httpx
import time

async def call_slm(question: str, question_id: str) -> InferenceResult:
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3:mini", "prompt": question, "stream": False},
            timeout=60.0,
        )
    latency_ms = (time.perf_counter() - start) * 1000
    data = resp.json()
    text = data["response"]
    return InferenceResult(
        model_class="slm",
        model_name="phi3:mini",
        question_id=question_id,
        output=text,
        latency_ms=latency_ms,
        input_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        cost_usd=0.0,  # amortize hardware separately — see checkpoint note
        cited_section=extract_section_citation(text),
    )
```

Add a resource-consumption profiler: peak RAM and tokens/second (`eval_count / eval_duration`). Run the full evaluation set cold, same as Lessons 1–2.

## 4. Checkpoint

Compute the SLM's citation-match rate and numeric-accuracy rate on the `jurisdiction_amendment` category specifically — expect this to be near zero. Compare against the general `definitional` category, where general building-code terminology (more likely to appear in pretraining data) should perform noticeably better. Model SLM cost as amortized compute (hardware cost / expected request volume), not literally $0 — a $0 line item misrepresents the real tradeoff in your dashboard.

**Next:** Lesson 4 adds multimodal routing for site-plan and diagram-bearing questions.

## References

1. Abdin, M. et al. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." arXiv:2404.14219. https://arxiv.org/abs/2404.14219
