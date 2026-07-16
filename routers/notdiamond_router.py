"""NotDiamond router (dry-run).

`model_router.select_model` returns NotDiamond's ranked model recommendation
WITHOUT running inference — a true dry-run selection. NotDiamond routes on query
complexity / cost / latency across a model catalog; it has NO grounding concept,
so grounded is always False (the finding, not a bug).

Requires the API key in the environment — either NOTDIAMOND_API_KEY or
NOT_DIAMOND_API_KEY (we pass it explicitly, so either spelling works). Candidate
models are a strong/cheap pair from NotDiamond's catalog; adjust to the current
catalog via `client.models.list()`.
"""

import os

from routers.base import RouterChoice, STRONG, CHEAP

# (provider, model) pairs from NotDiamond's catalog, split by tier. These are NOT
# our benchmark models (Opus 4.8 / Haiku 4.5 / phi3) — NotDiamond routes its own
# catalog, so we compare at the strength-tier level, not model-for-model.
# Verified against NotDiamond 1.7.0's catalog (probed via select_model — some
# model strings 400; these are accepted).
_STRONG = {("openai", "gpt-4o"), ("anthropic", "claude-3-7-sonnet-latest")}
_CHEAP = {("openai", "gpt-4o-mini"), ("anthropic", "claude-3-haiku-20240307")}

_client = None


def _get_client():
    global _client
    if _client is None:
        from notdiamond import NotDiamond
        key = os.environ.get("NOTDIAMOND_API_KEY") or os.environ.get("NOT_DIAMOND_API_KEY")
        _client = NotDiamond(api_key=key)
    return _client


def select(question: str, category: str) -> RouterChoice:
    providers = [{"provider": p, "model": m} for (p, m) in (_STRONG | _CHEAP)]
    resp = _get_client().model_router.select_model(
        llm_providers=providers,
        messages=[{"role": "user", "content": question}],
        tradeoff="cost",  # cost-aware routing; drop or change to compare tradeoffs
    )
    top = resp.providers[0]  # ranked; [0] is NotDiamond's pick
    picked = (top.provider, top.model)
    strength = STRONG if picked in _STRONG else CHEAP
    return RouterChoice("notdiamond", strength, grounded=False, raw_model=f"{top.provider}/{top.model}")
