# Lesson 1 — Foundation LLMs

## 1. Concept

A foundation model is the output of large-scale next-token pretraining on broad web-scale corpora, before any preference alignment is applied. In the standard training pipeline, the base model is refined afterward through supervised fine-tuning (SFT) on curated high-quality responses, then further aligned to human preference via reinforcement learning from human feedback (RLHF) or a direct-alignment method such as Direct Preference Optimization (DPO). The typical RLHF pipeline consists of three phases — supervised fine-tuning, preference/reward-model training, and reinforcement-learning optimization — and the SFT-only checkpoint is treated as an intermediate baseline, not yet fully aligned with human preference, that feeds into the later stages [1].

The classical RLHF alignment stage trains a reward model on human preference comparisons and then optimizes the policy against that reward model using Proximal Policy Optimization (PPO) [1]. DPO reparameterizes the reward function directly in terms of the policy's own parameters, eliminating the separate reward-model-training step while remaining mathematically equivalent to RLHF at the optimum [2].

**Why this matters immediately for a code-lookup assistant:** a raw foundation checkpoint has no reliable notion of "cite your source" or "say you don't know." Every commercial model you'll actually call (GPT-4-class, Claude, Gemini) is already instruction-tuned and aligned — what you're really evaluating in this lesson is the top capability tier's *parametric* knowledge of the Florida Building Code: what it got from pretraining data, with no retrieval assistance.

## 2. Why it matters for routing decisions

Parametric knowledge of a specific, frequently-amended code document is exactly the kind of narrow, fast-changing knowledge foundation models are worst at — code editions change every three years, and local amendments (Naples/Collier County) are far less represented in training data than the code itself. This lesson establishes your **baseline hallucination rate** for the flagship model with zero retrieval help. Expect it to be non-trivial. That number is the justification for Lesson 6.

## 3. Build increment

### 3.1 Configure your Anthropic API key

Before writing any client code, get a key and load it from a `.env` file so it never lives in source.

1. **Get a key.** Sign in at [console.anthropic.com](https://console.anthropic.com), open **Settings → API keys → Create Key**, and copy it (it looks like `sk-ant-...`). You only see the full value once — copy it now.
2. **Install the dependencies:**

   ```bash
   pip install anthropic python-dotenv
   ```
3. **Create a `.env` file** in your project root with the key on its own line:

   ```dotenv
   # .env
   ANTHROPIC_API_KEY=sk-ant-...your-key-here...
   ```
4. **Never commit it.** Add `.env` to `.gitignore`:

   ```gitignore
   .env
   ```
5. **Load it at startup.** `load_dotenv()` reads `.env` into `os.environ`. You do **not** pass the key into the client explicitly — `anthropic.Anthropic()` / `AsyncAnthropic()` reads `ANTHROPIC_API_KEY` from the environment automatically:

   ```python
   from dotenv import load_dotenv
   import anthropic

   load_dotenv()                       # .env  →  os.environ
   client = anthropic.AsyncAnthropic()  # picks up ANTHROPIC_API_KEY on its own
   ```

   If the key is missing or malformed, the first API call raises `anthropic.AuthenticationError` (401) — not a load-time error, so test one call before a full benchmark run.

### 3.2 Get the current input/output prices

Token prices change, and the pricing dict below is what turns raw token counts into the cost-per-answer numbers this whole benchmark is built on — so read them from the source, don't hard-code a half-remembered number:

- **Where to look:** the official pricing page, [platform.claude.com/docs/en/pricing](https://platform.claude.com/docs/en/pricing) (or the pricing panel in the Console). Prices are quoted as **USD per 1M (million) tokens**, with a separate **input** and **output** rate per model.
- **The Models API does _not_ return pricing** — `client.models.retrieve(...)` gives you the context window and capabilities, but you copy the two dollar figures from the pricing page by hand.
- **Current flagship rates** (re-verify before a run): Claude Opus 4.8 — **$5.00 in / $25.00 out** per 1M tokens. For reference, Sonnet 4.6 is $3.00/$15.00 and Haiku 4.5 is $1.00/$5.00.

Keep the pricing dict in the same per-1M-token units the page uses, so updating it is a straight copy of the two published numbers.

### 3.3 Implement the client

Implement `clients/foundation_client.py` and your first evaluation set:

```python
import time
import anthropic
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()                        # .env → os.environ (see 3.1)
client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

# The flagship / top-tier model this lesson benchmarks for parametric FBC knowledge.
FOUNDATION_MODEL = "claude-sonnet-5"

@dataclass
class InferenceResult:
    model_class: str
    model_name: str
    question_id: str
    output: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cited_section: str | None  # extracted via regex, e.g. "FBC 1006.2.1"

# USD per 1M tokens — copy these two numbers straight from
# https://platform.claude.com/docs/en/about-claude/pricing (re-check before each run).
MODEL_PRICING = {
    "claude-sonnet-5":  {"in": 2.00, "out": 10.00},  # flagship / foundation tier
}

def calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING[model_name]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000

async def call_foundation_model(question: str, question_id: str) -> InferenceResult:
    start = time.perf_counter()
    response = await client.messages.create(
        model=FOUNDATION_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": question}],
    )
    latency_ms = (time.perf_counter() - start) * 1000
    text = response.content[0].text
    usage = response.usage
    cost = calc_cost(FOUNDATION_MODEL, usage.input_tokens, usage.output_tokens)
    return InferenceResult(
        model_class="foundation",
        model_name=FOUNDATION_MODEL,
        question_id=question_id,
        output=text,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        cited_section=extract_section_citation(text),
    )
```

Build the evaluation set now (`data/fbc_eval_questions.csv`), 25–40 rows, each with: `question`, `gold_section`, `gold_answer`, `category` (definitional / numeric / jurisdiction-amendment). Example rows:

| question | gold_section | gold_answer | category |
|---|---|---|---|
| "What is the minimum clear width for a means of egress corridor?" | FBC 1020.2 | 44 inches (with exceptions for reduced-occupancy) | numeric |
| "Does the City of Naples require hurricane shutters or impact-rated glazing?" | Naples amendment to FBC Ch. 16 | impact-rated glazing or shutters, per wind-borne debris region | jurisdiction-amendment |

Run each question 3 times cold (no retrieval) against the foundation model and log whether `cited_section` matches `gold_section`.

## 4. Checkpoint

You should have `results/foundation_raw.jsonl` with per-question latency, cost, and a citation-match boolean. Compute the baseline hallucination rate: what percentage of answers cite a wrong or nonexistent section? This number is your headline finding for later lessons — write it down before you forget it.

**Next:** Lesson 2 adds the instruction-tuned client and a structured-output diff on numeric requirements.

## References

1. Wang, Z. et al. "A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More." arXiv:2407.16216. https://arxiv.org/abs/2407.16216
2. Rafailov, R. et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. https://arxiv.org/abs/2305.18290
