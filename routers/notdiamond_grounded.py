"""NotDiamond fed a GROUNDED prompt — the "leverage, don't rebuild" experiment.

Tests the orthogonal-layers hypothesis directly: if you do the grounding yourself
(retrieve the passage, inject it) and hand the *grounded* prompt to an off-the-shelf
router, does the router land on cheap+grounded — i.e. can you reach the custom
router's behavior by COMPOSING retrieval + NotDiamond, instead of building a bespoke
router?

The grounding decision is ours (we always inject the retrieved passage); NotDiamond
only makes the orthogonal model-tier choice on the already-grounded prompt. So
grounded is always True here; the open question the run answers is which TIER
NotDiamond picks once the answer is sitting in the context (cheap = the win).
"""

from routers.base import RouterChoice
from routers.notdiamond_router import _pick

_TEMPLATE = ("Answer using ONLY the building-code excerpt below. Cite the section.\n\n"
            "EXCERPT:\n{context}\n\nQUESTION: {question}")


def select(question: str, category: str, context: str | None = None) -> RouterChoice | None:
    if not context:
        return None  # ungroundable (q017 — a date, no code section); driver skips it
    strength, raw = _pick(_TEMPLATE.format(context=context, question=question))
    return RouterChoice("notdiamond_grounded", strength, grounded=True, raw_model=raw)
