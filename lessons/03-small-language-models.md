# Lesson 3 — Small Language Models (SLMs)

## 1. Concept

SLMs achieve disproportionate capability-per-parameter by prioritizing training-data quality over raw scale. Microsoft's Phi-3 family is the reference case: phi-3-mini, a 3.8 billion parameter model trained on 3.3 trillion tokens, has overall performance rivaling models such as Mixtral 8x7B and GPT-3.5 (69% on MMLU, 8.38 on MT-bench), despite being small enough to deploy on a phone [1]. This directly challenges naive scaling-law assumptions that hold data source fixed: the Phi-3 approach relies on a heavily filtered, curated training corpus plus synthetic data rather than raw scale, following the "textbooks are all you need" line of work [1]. Deployment implications are concrete: phi-3-mini can be quantized to 4 bits, occupying roughly 1.8GB of memory, and was demonstrated running fully offline on an iPhone 14 at over 12 tokens per second [1]. Parameter-scaled variants extend the family — phi-3-small (7B) and phi-3-medium (14B), trained for 4.8T tokens, reach 75% and 78% on MMLU respectively [1] — giving a capability dial within one family.

**Why this matters specifically for FBC lookups:** an SLM's pretraining corpus almost certainly under-represents the Florida Building Code relative to general web text, and it has essentially no exposure to Naples/Collier County local amendments — these are narrow, low-traffic municipal documents. Expect the SLM's cold (non-grounded) performance on jurisdiction-specific questions to be the weakest of all three model classes. That's the finding this lesson is designed to surface, not a flaw in your setup.

## 2. Why it matters for routing decisions

SLMs are the right default for high-volume, low-complexity tasks; latency-sensitive or on-device deployments (e.g., a field inspector's offline tablet app); and cost-sensitive pipelines. For a permitting-software vendor processing thousands of lookups a day, an SLM's near-zero marginal cost is compelling — *if* it can be paired with retrieval grounding to compensate for its narrow domain exposure. This lesson measures the gap that Lesson 4 will close.

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

Compute the SLM's citation-match rate on each category, and its resource profile (tokens/sec, latency). Model SLM cost as amortized compute (hardware cost / expected volume), not literally $0 — a $0 line item misrepresents the real tradeoff.

### Measured result (reference run — phi3:mini, CPU, no GPU)

Cold, no retrieval, 45 questions × 3 repeats, against the two hosted tiers:

| tier | correct citation | wrong (hallucination) | abstained | $/req | p50 latency |
|---|---|---|---|---|---|
| Opus 4.8 (foundation) | 26.5% | 31.8% | 41.7% | $0.0105 | 6.6 s |
| Haiku 4.5 (instruction-tuned) | 31.8% | 67.4% | 0.8% | $0.0004 | 0.8 s |
| **phi3:mini (SLM, local)** | **7.9%** | 39.4% | 52.8% | ~$0.001¹ | **68 s** |

¹ amortized wall-clock hardware cost (a documented placeholder), not $0 — see `slm_client.py`.

**The SLM is the weakest tier by a wide margin, exactly as predicted — and the *way* it's weak is the lesson.** Correct-citation recall collapses to **7.9%**, 3–4× below either hosted model, and near-floor in every category (jurisdiction 10.8%, numeric 6.7%, definitional and state_amendment 0%). A ~3.8B model simply has almost no parametric exposure to the Florida Building Code and none to Naples/Collier local amendments — there is nothing to recall.

Three findings worth carrying to the dashboard and the decision engine (Lesson 5):

- **"Free" is not free — it's slow.** At **68 s median latency (5.3 tokens/sec)** on CPU, phi3:mini is ~80× slower than Haiku. The near-zero *marginal* cost is real, but the latency makes it unusable for interactive lookups without a GPU, and the 3.7% timeout rate is a reliability cost a $0 line item hides. On the latency/cost chart it is a clear outlier.
- **The small model is honest, not reckless.** Like the flagship and unlike the schema-forced instruction-tuned model, phi3 **abstains more than it fabricates** (53% no-citation vs 39% wrong). It's reasonably calibrated about not knowing — it just doesn't know. Calibration without knowledge is still unusable.
- **This is the strongest case for grounding in the series.** An SLM that scores 7.9% cold but runs locally at near-zero marginal cost is the exact profile that Lesson 4 targets: if retrieval can lift *this* model's citation accuracy toward the hosted tiers, "cheap local model + good retrieval" becomes the compelling production architecture. That measurement is the whole point of Lesson 4.

**Next:** Lesson 4 adds retrieval grounding — the correct code passage injected as context — where hallucination rates should drop the most.

## References

1. Abdin, M. et al. "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone." arXiv:2404.14219. https://arxiv.org/abs/2404.14219
