"""
Small language model client (Lesson 3) — local Phi-3-mini via Ollama.

The point of this tier is the opposite of Lesson 1's flagship: a ~3.8B model
running on your own hardware, near-zero marginal cost, but almost no exposure to
the Florida Building Code and essentially none to Naples/Collier local
amendments. Expect cold jurisdiction_amendment performance to be the weakest of
all three classes — that gap is the finding, and it's what Lesson 4's grounding
is meant to close.

Prerequisite (this client hits a local server, so it must be running first):
    1. Install Ollama:  https://ollama.com/download
    2. Pull the model:  ollama pull phi3:mini      (~2.3 GB)
    3. Ollama serves on http://localhost:11434 automatically once installed.
If the server isn't up, every request is recorded as a failed row (a clean
ConnectError in the reliability metric) rather than crashing the run.
"""

import asyncio
import time
from dataclasses import dataclass

import httpx

from clients.citation_utils import extract_section_citation

SLM_MODEL = "phi3:mini"  # alt: "llama3.2:3b" — change here + pull it with ollama
OLLAMA_URL = "http://localhost:11434/api/generate"

# httpx raises these when the local server is down, slow, or errors — the harness
# records them as failed requests instead of aborting the run.
API_ERRORS = (httpx.HTTPError,)

# SLM cost is NOT $0 — you pay for the hardware's time. This amortized hourly rate
# (electricity + hardware depreciation for a modest inference box) is a PLACEHOLDER;
# treat it like the Section 4 assumptions banner and set it to your real figure.
# Cost is charged on wall-clock seconds, which is what owned compute actually bills.
AMORTIZED_HOURLY_USD = 0.05


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
    tokens_per_sec: float | None  # resource profile (Lesson 3): throughput


async def call_slm(question: str, question_id: str) -> InferenceResult:
    start = time.perf_counter()
    async with httpx.AsyncClient() as http:
        # Generous timeout: the FIRST request also loads the model into RAM,
        # which can take tens of seconds on a cold start.
        resp = await http.post(
            OLLAMA_URL,
            json={"model": SLM_MODEL, "prompt": question, "stream": False},
            timeout=180.0,
        )
        resp.raise_for_status()
    latency_ms = (time.perf_counter() - start) * 1000

    data = resp.json()
    text = data.get("response", "")
    output_tokens = data.get("eval_count", 0) or 0
    eval_ns = data.get("eval_duration", 0) or 0  # nanoseconds of generation
    tokens_per_sec = (output_tokens / (eval_ns / 1e9)) if eval_ns else None

    cost = (latency_ms / 1000) * (AMORTIZED_HOURLY_USD / 3600)
    return InferenceResult(
        model_class="slm",
        model_name=SLM_MODEL,
        question_id=question_id,
        output=text,
        latency_ms=latency_ms,
        input_tokens=data.get("prompt_eval_count", 0) or 0,
        output_tokens=output_tokens,
        cost_usd=cost,
        cited_section=extract_section_citation(text),
        tokens_per_sec=tokens_per_sec,
    )
