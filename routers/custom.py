"""Custom, domain-aware router — the readable, extensible lookup table.

Encodes the Lessons 1-4 findings: GROUND everything groundable, then pick the
CHEAPEST tier that clears an accuracy floor. This is the opposite of an off-the-
shelf router's embedded ranking — the decision is a table you can read and extend.

Numbers are the MEASURED grounded citation-match rates by category (see FINDINGS.md
/ results/*_grounded.jsonl). Update them if you re-run the benchmark.
"""

from routers.base import RouterChoice, STRONG, CHEAP

# Grounded citation-match rate by (tier, category), from the measured grounded runs.
# "cheap_local" = phi3:mini (cheapest, offline), "cheap" = Haiku 4.5 (schema), "strong" = Opus 4.8.
GROUNDED_ACC = {
    "cheap_local": {"jurisdiction_amendment": 0.70, "numeric": 0.47, "state_amendment": 0.00, "definitional": 0.00},
    "cheap":       {"jurisdiction_amendment": 0.82, "numeric": 0.75, "state_amendment": 0.67, "definitional": 0.67},
    "strong":      {"jurisdiction_amendment": 0.64, "numeric": 0.63, "state_amendment": 0.67, "definitional": 0.00},
}
# Categories where the correct passage can be retrieved (q017 is a FEMA date -> not groundable).
GROUNDABLE = {"jurisdiction_amendment", "numeric", "state_amendment", "definitional"}
FLOOR = 0.50  # minimum grounded correct-rate to keep a query on a cheap tier

# Tiers ordered cheapest-first; the router takes the first that clears the floor when grounded.
_CHEAPEST_FIRST = ["cheap_local", "cheap", "strong"]


def select(question: str, category: str, context: str | None = None) -> RouterChoice:
    grounded = category in GROUNDABLE
    if not grounded:
        # No passage to ground on — fall back to the strongest parametric memory.
        return RouterChoice("custom", STRONG, grounded=False, raw_model="strong (ungroundable)")

    for tier in _CHEAPEST_FIRST:
        if GROUNDED_ACC[tier].get(category, 0.0) >= FLOOR:
            strength = STRONG if tier == "strong" else CHEAP
            return RouterChoice("custom", strength, grounded=True, raw_model=f"{tier}+grounded")
    # Nothing clears the floor even grounded -> strongest, still grounded (best available).
    return RouterChoice("custom", STRONG, grounded=True, raw_model="strong+grounded")
