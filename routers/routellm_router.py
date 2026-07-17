"""RouteLLM router (dry-run) — runs inside the Docker image only.

RouteLLM's ML stack (litellm/torch/transformers) won't install on the Windows
host, so this module is imported opportunistically: the driver skips it when
`routellm` isn't available and includes it when run in docker/Dockerfile.

Dry-run = RouteLLM's `bert` router scores P(strong model meaningfully better)
per prompt; compare to a threshold to get strong/weak. No target LLM is called.
(The `mf` router was the first choice but it embeds each prompt via the OpenAI
API — a real network call needing a real key; `bert` is a fully local fine-tuned
classifier, so the dry-run stays offline and reproducible.) RouteLLM only knows a
strong/weak pair and has NO grounding concept, so grounded is always False — the
finding, not a limitation to fix.
"""

import pandas as pd

from routers.base import RouterChoice, STRONG, CHEAP

# strong/weak are labels only (never called). The mf router's win-rate is
# model-agnostic; these just name the tiers.
_ctrl = None
_THRESHOLD = 0.5  # win-rate >= threshold -> strong. Calibrate for a target
#                    strong-call fraction if you want a specific split.


def _controller():
    global _ctrl
    if _ctrl is None:
        from routellm.controller import Controller
        _ctrl = Controller(
            routers=["bert"],
            strong_model="gpt-4-1106-preview",
            weak_model="mixtral-8x7b-instruct-v0.1",
        )
    return _ctrl


def select(question: str, category: str, context: str | None = None) -> RouterChoice:
    # VERIFY against the current RouteLLM API if this errors; the intent is a
    # per-prompt routing decision without inference.
    wr = _controller().batch_calculate_win_rate(pd.Series([question]), router="bert").iloc[0]
    strength = STRONG if wr >= _THRESHOLD else CHEAP
    return RouterChoice("routellm", strength, grounded=False, raw_model=f"bert/win_rate={wr:.2f}")
