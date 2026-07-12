"""
Foundation-tier client (Lesson 1) — Google Gemini.

Provider choice: Gemini's free tier covers the full benchmark at $0 while the
Anthropic account is unfunded. The harness (run_benchmark.py) and the
InferenceResult schema are provider-agnostic — swapping providers means
changing only this module.

cost_usd records LIST prices (pay-as-you-go rates) even though the free-tier
run itself costs $0 — the benchmark's cost dimension compares model classes,
and $0 rows would misrepresent that tradeoff (same reasoning as the SLM
amortized-cost rule in Lesson 3).
"""

import asyncio
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from clients.citation_utils import extract_section_citation

load_dotenv()
client = genai.Client()  # reads GEMINI_API_KEY from the environment

# Free-tier quota reality (measured 2026-07-10, error-message quotaValue):
#   pro-class models        -> 0 RPD (no free tier at all)
#   gemini-3.5-flash        -> 20 RPD (too small for a 135-request run)
#   gemini-3-flash-preview  -> separate quota bucket, depth measured empirically
FOUNDATION_MODEL = "gemini-3-flash-preview"

# Exceptions the harness records as failed requests (reliability metric)
# instead of crashing the run.
API_ERRORS = (errors.APIError,)


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


# USD per 1M tokens, list prices — verify at https://ai.google.dev/gemini-api/docs/pricing
# before each run (Lesson 1 §3.2).
MODEL_PRICING = {
    "gemini-3.5-flash":       {"in": 1.50, "out": 9.00},  # checked 2026-07-10
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},  # TODO VERIFY on the pricing page — unconfirmed estimate
}


def calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING[model_name]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000


async def call_foundation_model(question: str, question_id: str) -> InferenceResult:
    # Free-tier RPM limits surface as 429s mid-run; those are quota pacing,
    # not model unreliability, so retry them with backoff instead of letting
    # them pollute the failure-rate metric. Everything else propagates.
    for attempt in range(4):
        start = time.perf_counter()  # restart per attempt so latency excludes backoff sleeps
        try:
            response = await client.aio.models.generate_content(
                model=FOUNDATION_MODEL,
                contents=question,
                # Thinking tokens count against this cap and can exceed 1K on
                # their own — too low a cap yields empty text (observed at 100).
                config=types.GenerateContentConfig(max_output_tokens=4096),
            )
            break
        except errors.APIError as e:
            if getattr(e, "code", None) == 429 and attempt < 3:
                await asyncio.sleep(20 * (attempt + 1))
            else:
                raise
    latency_ms = (time.perf_counter() - start) * 1000

    text = response.text or ""
    um = response.usage_metadata
    input_tokens = um.prompt_token_count or 0
    # Thinking tokens bill as output tokens on Gemini 2.5+/3.x models.
    output_tokens = (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
    cost = calc_cost(FOUNDATION_MODEL, input_tokens, output_tokens)
    return InferenceResult(
        model_class="foundation",
        model_name=FOUNDATION_MODEL,
        question_id=question_id,
        output=text,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        cited_section=extract_section_citation(text),
    )
