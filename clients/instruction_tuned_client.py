"""
Instruction-tuned client (Lesson 2) — Claude Haiku 4.5.

Structurally identical to the foundation client, but pointed at a cheaper
mid-tier instruction-tuned model AND holding it to a rigid JSON output
contract. The thing being tested is instruction-following fidelity: does a
cheap model return schema-valid structured output as reliably as the flagship,
on the numeric and jurisdiction_amendment questions a permitting integration
would actually consume?

The prompt asks for JSON and we validate it with Pydantic *after the fact* —
deliberately NOT using the API's strict structured-output mode. Strict mode
would guarantee valid JSON and thereby erase the very signal we want:
`schema_valid` measures whether the model complies with a schema stated in the
prompt, which is what a real integration depends on.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from clients.citation_utils import extract_section_citation, normalize_section

load_dotenv()
client = anthropic.AsyncAnthropic()

INSTRUCTION_TUNED_MODEL = "claude-haiku-4-5"

API_ERRORS = (anthropic.APIError,)

# USD per 1M tokens, list prices — verify before each run (Lesson 1 §3.2).
MODEL_PRICING = {
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},  # checked 2026-07-12
}

SCHEMA_PROMPT = """Answer the Florida Building Code question below.
Respond ONLY with valid JSON matching this schema — no prose, no markdown fences:
{{"value": <number or null>, "unit": <string or null>, "section": <string>, "confidence": <"high" or "low">}}

Question: {question}
"""


class AnswerSchema(BaseModel):
    value: float | None = None
    unit: str | None = None
    section: str
    confidence: Literal["high", "low"]


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
    schema_valid: bool  # did the model honor the JSON contract? (Lesson 2 metric)


def calc_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    p = MODEL_PRICING[model_name]
    return (input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000


def validate_json_schema(text: str) -> AnswerSchema | None:
    """Parse `text` as the required schema. Returns the validated model, or None
    if the model broke the contract (non-JSON, missing/extra fields, wrong type).
    Tolerates a ```json fence since that's a compliance near-miss, not the model
    inventing a citation — but a bare prose answer fails, as it should."""
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    try:
        return AnswerSchema.model_validate_json(candidate)
    except ValidationError:
        return None


async def call_instruction_tuned(question: str, question_id: str) -> InferenceResult:
    for attempt in range(4):
        start = time.perf_counter()
        try:
            response = await client.messages.create(
                model=INSTRUCTION_TUNED_MODEL,
                max_tokens=400,
                messages=[{"role": "user", "content": SCHEMA_PROMPT.format(question=question)}],
            )
            break
        except anthropic.RateLimitError:
            if attempt < 3:
                await asyncio.sleep(20 * (attempt + 1))
            else:
                raise
    latency_ms = (time.perf_counter() - start) * 1000

    text = "".join(block.text for block in response.content if block.type == "text")
    parsed = validate_json_schema(text)

    # Cite from the structured `section` field when the contract held; otherwise
    # fall back to scanning the raw text, so a schema break still gets a fair
    # citation-match shot (and the two failures — bad schema, bad citation —
    # stay independently measurable). The `section` field is a dedicated,
    # already-isolated citation, so normalize it directly if the prose extractor
    # (which needs a prefix) doesn't catch it — a bare "1003.2" is valid there.
    #
    # Note: the schema FORCES the model to name a section, so unlike the
    # free-prose foundation client, abstention ("no citation") is largely
    # unavailable here — a Lesson 2 finding in itself. confidence="low" is the
    # only honesty channel the contract leaves open.
    if parsed is not None:
        cited_section = extract_section_citation(parsed.section) or normalize_section(parsed.section)
    else:
        cited_section = extract_section_citation(text)

    usage = response.usage
    cost = calc_cost(INSTRUCTION_TUNED_MODEL, usage.input_tokens, usage.output_tokens)
    return InferenceResult(
        model_class="instruction_tuned",
        model_name=INSTRUCTION_TUNED_MODEL,
        question_id=question_id,
        output=text,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost,
        cited_section=cited_section,
        schema_valid=parsed is not None,
    )
