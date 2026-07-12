# Lesson 1 — Foundation LLMs: The Cold Parametric Baseline

> **What you're actually measuring.** No commercial API exposes a true foundation model — a raw pretrained checkpoint with no instruction tuning or alignment. Claude, GPT, and Gemini are all already tuned. So this lesson benchmarks the strongest model in your lineup answering **cold, from parametric memory alone** — no retrieval, no supplied code text. `model_class="foundation"` in the code names this *role in the routing architecture* (the trust-the-model's-memory tier), not a literal base checkpoint. That makes every failure recorded here a lower bound: this is the best case for trusting model memory, and Lessons 2–6 measure everything as a delta from it.

## 1. Concept

A foundation model is the output of large-scale next-token pretraining on broad web-scale corpora, before any preference alignment is applied. In the standard training pipeline, the base model is refined afterward through supervised fine-tuning (SFT) on curated high-quality responses, then further aligned to human preference via reinforcement learning from human feedback (RLHF) or a direct-alignment method such as Direct Preference Optimization (DPO). The typical RLHF pipeline consists of three phases — supervised fine-tuning, preference/reward-model training, and reinforcement-learning optimization — and the SFT-only checkpoint is treated as an intermediate baseline, not yet fully aligned with human preference, that feeds into the later stages [1].

The classical RLHF alignment stage trains a reward model on human preference comparisons and then optimizes the policy against that reward model using Proximal Policy Optimization (PPO) [1]. DPO reparameterizes the reward function directly in terms of the policy's own parameters, eliminating the separate reward-model-training step while remaining mathematically equivalent to RLHF at the optimum [2].

**Why this matters immediately for a code-lookup assistant:** a raw foundation checkpoint has no reliable notion of "cite your source" or "say you don't know." Every commercial model you'll actually call (GPT-4-class, Claude, Gemini) is already instruction-tuned and aligned — what you're really evaluating in this lesson is the top capability tier's _parametric_ knowledge of the Florida Building Code: what it got from pretraining data, with no retrieval assistance.

## 2. Why it matters for routing decisions

Parametric knowledge of a specific, frequently-amended code document is exactly the kind of narrow, fast-changing knowledge foundation models are worst at — code editions change every three years, and local amendments (Naples/Collier County) are far less represented in training data than the code itself. This lesson establishes your **baseline hallucination rate** for your foundation-tier model with zero retrieval help. Expect it to be non-trivial. That number is the justification for Lesson 6.

## 3. Build increment

> **Provider note.** The walkthrough below uses Anthropic as the worked example. The repo's actual client (`clients/foundation_client.py`) targets **Google Gemini's free tier** instead — same harness, same `InferenceResult` schema, different ~20 lines of SDK calls — because the free tier covers the whole benchmark at $0. The Gemini equivalents: get a key at [aistudio.google.com](https://aistudio.google.com) → put it in `.env` as `GEMINI_API_KEY=...` → `pip install google-genai python-dotenv` → `genai.Client()` reads the key from the environment automatically → prices live at [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing). Two Gemini-specific gotchas the client handles: free-tier 429s are quota pacing (retried with backoff, not counted as failures), and thinking tokens count against `max_output_tokens` *and* bill as output tokens — set the cap generously and include `thoughts_token_count` in cost math. Everything else in this section applies to either provider.

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

- **Where to look:** the official pricing page, [platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing) (or the pricing panel in the Console). Prices are quoted as **USD per 1M (million) tokens**, with a separate **input** and **output** rate per model.
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

from clients.citation_utils import extract_section_citation

load_dotenv()                        # .env → os.environ (see 3.1)
client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

# The flagship / top-tier model this lesson benchmarks for parametric FBC knowledge.
FOUNDATION_MODEL = "claude-sonnet-4-6"

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
    "claude-sonnet-4-6":  {"in": 3.00, "out": 15.00}  # foundation-tier role (strongest model in the lineup)
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

Build the evaluation set now (`data/fbc_eval_questions.csv`), 45 rows, each with: `question_id`, `question`, `gold_section`, `gold_answer`, and `category` (definitional / numeric / state_amendment / jurisdiction_amendment — the last two separate Florida-statewide requirements from Collier/Naples local code, the two tiers of "narrow knowledge" this benchmark stresses). Example rows:

| question_id | question                                                                                                                                                  | gold_section                                  | gold_answer                                                                                                                                                                 | category               |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| q001        | "What section of the 2023 Florida Building Code, Building, governs the width and capacity of corridors used as a means of egress?"                        | FBC Building 1020.2                           | Section 1020.2 of Chapter 10 (Means of Egress) sets corridor width and capacity requirements for the 2023 8th Edition Florida Building Code, Building.                      | numeric                |
| q015        | "Which Collier County code section governs flood variance requests, including deviations from the Base Flood Elevation plus local freeboard requirement?" | Collier County Code of Ordinances, Sec. 62-77 | Sec. 62-77 (Chapter 62, Floods) governs flood variance requests, including BFE-plus-freeboard deviation requests reviewed by the Building Board of Adjustments and Appeals. | jurisdiction_amendment |

Every `gold_section` in the starter CSV was confirmed directly against the published 2023 FBC 8th Edition and the Collier County / City of Naples code at the time of writing. Building codes are amended on a roughly three-year cycle — if you're picking this up later or benchmarking against a different jurisdiction, re-verify each row against the current adopted edition before running the harness, since a stale `gold_section` silently corrupts every downstream accuracy number.

Run each question 3 times cold (no retrieval) against the foundation model via the shared harness, which writes one JSON line per request and prints the baseline summary:

```bash
python run_benchmark.py                    # 45 questions x 3 repeats
python run_benchmark.py --repeats 1 --limit 2   # cheap smoke test first
```

The harness normalizes `gold_section` with the same regex extractor used on model output before comparing (gold values are prose-styled, e.g. "FBC Building 1020.2", while `cited_section` is normalized to "1020.2" — a raw string equality would score nearly everything as a miss).

`extract_section_citation()` lives in `clients/citation_utils.py` — a shared regex-based utility imported by every client in this series (foundation, instruction-tuned, SLM, multimodal), so you only implement it once here. It's deliberately regex-based rather than LLM-based: the citation-match metric needs to be a deterministic signal independent of any judge model, otherwise you're measuring the judge's reliability instead of the model under test.

## 4. Checkpoint

You should have `results/foundation_raw.jsonl` with per-question latency, cost, and a citation-match boolean. Compute the baseline citation outcomes: of the answers with a scoreable gold section, what fraction cite the **correct** section, a **wrong/invented** section (hallucination), or **no** section (abstention)? Split wrong-vs-none deliberately — for a compliance tool, a fabricated citation is far more dangerous than an honest "I don't know."

### Measured baseline (reference run)

Cold, no retrieval, 45 questions × 3 repeats. Two model tiers on the identical question set:

| outcome | Claude Opus 4.8 (flagship) | Gemini-3-flash (free tier) |
|---|---|---|
| correct citation | 26.5% | 38.6% |
| **wrong / invented (hallucination)** | **31.8%** | **52.3%** |
| declined to cite (abstention) | 41.7% | 9.1% |
| scoreable n | 132 | 44 |

**The headline is the calibration contrast, not a single hallucination number.** Paying ~10× more per token did *not* meaningfully improve Florida-code recall — both tiers cite the right section on only a quarter to a third of questions, and both are worst on the Collier/Naples `jurisdiction_amendment` category, exactly as Section 2 predicted. What changes is the **failure mode**: the cheaper model fills its knowledge gap by *inventing* a plausible citation, while the flagship fills it by *declining to answer* — often verbatim, e.g. *"I don't have Section 202 of the 2023 Florida Building Code memorized with enough precision."* That is the alignment story from Section 1 showing up in the data: RLHF/DPO did not add code knowledge, it added calibrated honesty about the absence of it.

Two methodological notes worth carrying forward:

- **Repeats earn their cost through instability, not independence.** 13 of 44 questions flipped between abstaining and citing across the three runs — nondeterministic recall a single pass would hide. (They do *not* add statistical independence: three calls to frozen weights are highly correlated, so effective n ≈ questions, not questions × repeats.)
- **A 0% category match can be good behavior.** The flagship scores 0% on `definitional` not by fabricating but by abstaining on §202 — inspect `results/hallucinated_citations.csv` before quoting any category number.

The practical conclusion is the same for both tiers and sets up the rest of the series: **cold parametric answering is unusable for code compliance — abstention is merely safer than fabrication, not correct.** The delta that grounding buys (Lesson 6) is measured against exactly this baseline.

**Next:** Lesson 2 adds the instruction-tuned client and a structured-output diff on numeric requirements.

## References

1. Wang, Z. et al. "A Comprehensive Survey of LLM Alignment Techniques: RLHF, RLAIF, PPO, DPO and More." arXiv:2407.16216. https://arxiv.org/abs/2407.16216
2. Rafailov, R. et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. https://arxiv.org/abs/2305.18290
