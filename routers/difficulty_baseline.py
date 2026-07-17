"""Difficulty-heuristic router (dry-run) — ILLUSTRATIVE, not evidence.

The evidence in Lesson 7 is the two *real* routers (RouteLLM in Docker, NotDiamond
via its API), both of which land 100% ungrounded. This module's job is different:
it's a transparent, readable stand-in so you can *see* the mechanism behind that
result — "difficulty → model tier" simply has no grounding variable to set.

It routes on estimated query difficulty → tier, which is how real difficulty
routers behave. Its empty grounded band is NOT independent corroboration — it is
true *by construction* (grounded is hardcoded False below), so don't cite it as
proof. Cite it as the legible explanation of why a tier-only framing can't express
the grounding decision, while the real routers show it holds in the wild.
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


def select(question: str, category: str, context: str | None = None) -> RouterChoice:
    strength = STRONG if _difficulty(question) >= _THRESHOLD else CHEAP
    return RouterChoice("difficulty_baseline", strength, grounded=False,
                        raw_model=f"{strength} (difficulty={_difficulty(question):.1f})")
