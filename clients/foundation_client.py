"""
Foundation-tier client (Lesson 1) — Anthropic Claude Opus 4.8 (flagship).

This is the canonical foundation-tier run: the strongest widely available
model answering cold, from parametric memory alone. A free-tier Gemini variant
is preserved in foundation_client_gemini.py (results in
results/foundation_gemini3flash_raw.jsonl) for a flash-vs-flagship comparison.

The harness (run_benchmark.py) and the InferenceResult schema are
provider-agnostic — swapping providers means changing only this module.
"""

import asyncio
import time
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from clients.citation_utils import extract_section_citation

load_dotenv()
client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from the environment

FOUNDATION_MODEL = "claude-opus-4-8"

# Exceptions the harness records as failed requests (reliability metric)
# instead of crashing the run.
API_ERRORS = (anthropic.APIError,)


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
    cited_section: str | None


# USD per 1M tokens, list prices — verify at https://platform.claude.com/docs/en/about-claude/pricing
# before each run (Lesson 1 §3.2).
MODEL_PRICING = {
    "claude-opus-4-8":  {"in": 5.00, "out": 25.00},  # checked 2026-07-12
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
}


def calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING[model_name]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000


async def call_foundation_model(question: str, question_id: str) -> InferenceResult:
    # Tier-1 accounts have modest per-minute token limits; 429s are quota
    # pacing, not model unreliability — retry them with backoff (on top of the
    # SDK's own retries) instead of polluting the failure-rate metric.
    for attempt in range(4):
        start = time.perf_counter()  # restart per attempt so latency excludes backoff sleeps
        try:
            response = await client.messages.create(
                model=FOUNDATION_MODEL,
                max_tokens=800,
                messages=[{"role": "user", "content": question}],
            )
            break
        except anthropic.RateLimitError:
            if attempt < 3:
                await asyncio.sleep(20 * (attempt + 1))
            else:
                raise
    latency_ms = (time.perf_counter() - start) * 1000

    text = "".join(block.text for block in response.content if block.type == "text")
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
