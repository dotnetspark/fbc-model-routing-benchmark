# Lesson 2 — Instruction-Tuned Models

## 1. Concept

Instruction tuning (IT) is supervised fine-tuning on (instruction, response) pairs, distinct in purpose from preference alignment: IT teaches format-following and task-following behavior, while alignment (RLHF/DPO) corrects values and refusal behavior. Instruction tuning on a well-established task-specific instruction set has been shown to inject domain-specific knowledge into general LLMs, transferring them into domain experts (e.g., in medicine, law, and finance), whereas alignment tuning separately corrects unexpected model behaviors to match human values and preferences [1]. Production-grade instruction-tuned models stack both, but research shows the stacking need not be strictly sequential — a unified fine-tuning approach (UFT) trained solely on instruction data, using a generalized implicit reward function, has been shown to outperform plain SFT by minimizing divergence from the pretrained model [2].

For this project, the practical distinction you're testing is **instruction-following fidelity under a rigid output contract** — something a code-compliance tool depends on completely. A model that answers "44 inches, but there are exceptions" in prose is useless to a permitting-software integration; you need `{"value": 44, "unit": "inches", "exceptions": [...], "section": "1020.2"}` every time.

## 2. Why it matters for routing decisions

Instruction-tuned models are your default for any question requiring structured output — which is most of what a building-code assistant needs to return to a downstream system (a permit-checking form, a contractor's estimating tool). The key risk you're testing: does a cheaper mid-tier instruction-tuned model hold the schema contract as reliably as the flagship foundation-tier model on numeric and jurisdiction_amendment questions?

## 3. Build increment

Add `clients/instruction_tuned_client.py`, structurally identical to Lesson 1's client but pointed at a mid-tier instruction-tuned model (GPT-4o-mini, Claude Haiku-class, or Llama-3.1-8B-Instruct), and require structured JSON output:

```python
SCHEMA_PROMPT = """Answer the Florida Building Code question below.
Respond ONLY with valid JSON matching this schema:
{{"value": <number or null>, "unit": <string or null>, "section": <string>, "confidence": <"high"|"low">}}

Question: {question}
"""

async def call_instruction_tuned(question: str, question_id: str, client) -> InferenceResult:
    start = time.perf_counter()
    response = await client.messages.create(
        model="your-instruction-tuned-model",
        max_tokens=400,
        messages=[{"role": "user", "content": SCHEMA_PROMPT.format(question=question)}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text = response.content[0].text
    schema_valid = validate_json_schema(text)  # Pydantic model check
    # ... construct InferenceResult, tag model_class="instruction_tuned",
    #     add a schema_valid field
```

Run this against your full evaluation set, focusing on the `numeric` and `jurisdiction_amendment` categories. Add a schema-validation-failure counter alongside your existing citation-match check.

## 4. Checkpoint

You should have a second raw result file plus a diff report comparing foundation vs. instruction-tuned on: (a) schema-validity rate, (b) numeric-answer accuracy, (c) citation-match rate, (d) cost and latency.

### Measured result (reference run)

Cold, no retrieval, 45 questions × 3 repeats. Claude Haiku 4.5 under the JSON schema contract, against the Lesson 1 flagship baseline:

| metric | Opus 4.8 (free prose) | Haiku 4.5 (forced JSON schema) |
|---|---|---|
| schema-valid rate | — (unconstrained) | **99.3%** |
| correct citation | 26.5% | 31.8% |
| **wrong / invented (hallucination)** | 31.8% | **67.4%** |
| declined to cite (abstention) | 41.7% | **0.8%** |
| cost / request | $0.01046 | **$0.00036** (~29× cheaper) |
| p50 latency | 6,632 ms | **840 ms** (~8× faster) |

**The headline is the tradeoff, not the win.** The cheap model holds the schema contract almost perfectly (99.3% valid JSON) at ~29× lower cost and ~8× lower latency — and its *correct*-citation rate is actually a touch **higher** than the flagship's. So structure didn't cost accuracy. What it cost is the model's safest failure mode: the schema's `section` field is **required**, so the model can never answer "I don't know." Abstention collapses from 42% to under 1%, and nearly all of that mass reappears as **wrong citations** — hallucination roughly doubles (31.8% → 67.4%).

This is the counterintuitive result to flag for the dashboard, and it generalizes past this project: **a schema guarantees a *parseable* answer, not a *correct* one — and by forbidding "I'm not sure," it actively converts calibrated uncertainty into confident fabrication.** For a permitting integration that consumes the JSON downstream, that is exactly the risk to design around: the field will always be populated, so a `confidence` field (or a grounding step, Lesson 4) is doing the real safety work, not the schema.

Note the metric subtlety carried over from Lesson 1: because the forced schema removes abstention, the instruction-tuned client's "no citation" rate is ~0 by construction — the correct comparison is *wrong-vs-correct*, not *hallucination-vs-abstention*. The `confidence: "low"` field is the only honesty channel the contract leaves open; a natural Lesson 2 extension is to score whether the model sets it when it's about to be wrong.

**Next:** Lesson 3 adds a local SLM and tests whether it can handle FBC lookups at all without help.

## References

1. Wang, Z. et al. "A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More." arXiv:2407.16216. https://arxiv.org/abs/2407.16216
2. Wang, Z. et al. "UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function." arXiv:2410.21438. https://arxiv.org/abs/2410.21438
