"""NotDiamond router (dry-run).

`model_router.select_model` returns NotDiamond's ranked model recommendation
WITHOUT running inference — a true dry-run selection. NotDiamond routes on query
complexity / cost / latency across a model catalog; it has NO grounding concept,
so grounded is always False (the finding, not a bug).

Requires NOT_DIAMOND_API_KEY in the environment (note the underscores — the SDK
uses NOT_DIAMOND_API_KEY). Candidate models are a strong/cheap pair from
NotDiamond's catalog; adjust to the current catalog via `client.models.list()`.
"""

from routers.base import RouterChoice, STRONG, CHEAP

# (provider, model) pairs from NotDiamond's catalog, split by tier. These are NOT
# our benchmark models (Opus 4.8 / Haiku 4.5 / phi3) — NotDiamond routes its own
# catalog, so we compare at the strength-tier level, not model-for-model.
_STRONG = {("openai", "gpt-4o"), ("anthropic", "claude-3-5-sonnet-20241022")}
_CHEAP = {("openai", "gpt-4o-mini"), ("anthropic", "claude-3-5-haiku-20241022")}

_client = None


def _get_client():
    global _client
    if _client is None:
        from notdiamond import NotDiamond
        _client = NotDiamond()  # reads NOT_DIAMOND_API_KEY from env
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
