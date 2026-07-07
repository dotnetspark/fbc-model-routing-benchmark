# Lesson 5 — Fine-Tuned Models

## 1. Concept

Fine-tuning takes a pretrained (often already instruction-tuned) model and further trains it on a narrow, task-specific dataset to shift its behavior distribution toward a domain. This is mechanically distinct from prompting: prompting conditions behavior at inference time within a fixed weight space, while fine-tuning permanently updates weights (or, in parameter-efficient variants like LoRA/QLoRA, a small adapter on frozen weights). The research consensus treats fine-tuning and alignment as points on the same pipeline — the standard RLHF pipeline itself starts with a pretrained model refined via supervised fine-tuning on curated high-quality responses, producing the baseline for the later alignment stages [1]. Domain-specialization instruction tuning specifically has been shown to transfer general-purpose LLMs into narrow experts across fields such as medicine, law, and finance via task-specific instruction sets [1] — the Florida Building Code is a direct analog: a narrow, stable, highly-structured document corpus.

## 2. Why it matters for routing decisions

Fine-tuning an SLM on the full text of the Florida Building Code plus Naples/Collier amendments is justified when: (a) grounding via RAG (Lesson 6) still leaves a quality gap because the model can't reliably parse cross-references and exceptions even with context supplied, (b) query volume is high enough to amortize training cost against per-request savings versus a foundation model, or (c) you need the assistant to work fully offline (e.g., a field-inspector tablet app with no connectivity) where RAG retrieval infrastructure isn't available. This lesson does not fine-tune anything — it builds the calculator that tells you when it becomes worth doing, using your own measured numbers from Lessons 1–3.

## 3. Build increment

Add `analysis/finetune_breakeven.py`:

```python
def breakeven_analysis(
    large_model_cost_per_req: float,
    finetune_one_time_cost: float,
    finetuned_model_cost_per_req: float,
    monthly_requests: int,
) -> dict:
    months = list(range(1, 13))
    cumulative_large = [large_model_cost_per_req * monthly_requests * m for m in months]
    cumulative_finetuned = [
        finetune_one_time_cost + finetuned_model_cost_per_req * monthly_requests * m
        for m in months
    ]
    breakeven_month = next(
        (m for m, (l, f) in zip(months, zip(cumulative_large, cumulative_finetuned)) if f < l),
        None,
    )
    return {
        "breakeven_month": breakeven_month,
        "cumulative_large": cumulative_large,
        "cumulative_finetuned": cumulative_finetuned,
    }
```

Model a realistic scenario: a permitting-software vendor processing 50,000 code-lookup requests/month. Feed the calculator your Lesson 1 foundation-model cost-per-request as the baseline, and a plausible fine-tuning cost estimate (data curation on ~500 curated FBC Q&A pairs + QLoRA training run on a 3–8B model) as the alternative. Plot the breakeven curve as a dashboard panel.

## 4. Checkpoint

You have a reusable breakeven calculator, parameterized by your own measured costs, that outputs a specific request-volume threshold rather than an unfalsifiable claim like "fine-tuning helps at scale." This is the artifact that makes the analysis credible to a technical or investor audience.

**Next:** Lesson 6 adds grounding via retrieved code text — this is where hallucination rates should drop the most.

## References

1. Wang, Z. et al. "A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More." arXiv:2407.16216. https://arxiv.org/abs/2407.16216
