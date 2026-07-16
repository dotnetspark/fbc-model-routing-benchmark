"""Difficulty-heuristic router (dry-run) — stand-in for a generic difficulty router.

Substituted for RouteLLM, whose ML dependency stack (litellm/torch/transformers/
datasets/pyarrow/huggingface-hub) has unresolvable version conflicts on this
Windows box — the exact install landmine LESSON7_PLAN.md §11 anticipated.

It routes on estimated query difficulty → model tier, which is how real difficulty
routers (RouteLLM et al.) behave, and — crucially — it has NO grounding concept,
so grounded is always False. That empty-grounded-band behavior is the shared
property being demonstrated; the specific strong/cheap split is illustrative.

Transparent by construction (a readable heuristic), which is the point: unlike a
trained/closed router, you can see and change its logic — but it still can't
express the grounding decision, because "difficulty → tier" has no such axis.
"""

from routers.base import RouterChoice, STRONG, CHEAP

# Markers that make an FBC question "harder" for a weak model: cross-references,
# specific numeric/standard lookups, multi-condition reasoning.
_MARKERS = (
    "table", "figure", "calculate", "factor", "capacity", "which standard",
    "conditions", "per occupant", "risk category", "ultimate design",
    "geotechnical", "hydrostatic", "amended", "ordinance",
)
_THRESHOLD = 2.0  # score >= threshold -> strong; tuned to split the set, not to win


def _difficulty(question: str) -> float:
    q = question.lower()
    return len(question) / 90.0 + sum(m in q for m in _MARKERS)


def select(question: str, category: str) -> RouterChoice:
    strength = STRONG if _difficulty(question) >= _THRESHOLD else CHEAP
    return RouterChoice("difficulty_baseline", strength, grounded=False,
                        raw_model=f"{strength} (difficulty={_difficulty(question):.1f})")
