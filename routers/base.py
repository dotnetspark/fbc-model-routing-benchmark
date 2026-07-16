"""Canonical dry-run router interface.

Routers use different model catalogs (RouteLLM: a strong/weak pair; NotDiamond:
its own list; ours: Opus/Haiku/phi3). So we never compare exact model names — every
router's pick is collapsed to a `strength` tier plus a `grounded` flag. Only a
router that can *decide to ground* ever sets grounded=True; every off-the-shelf
tier-router is structurally stuck at grounded=False, which is the finding.
"""

from dataclasses import dataclass

STRONG = "strong"
CHEAP = "cheap"


@dataclass
class RouterChoice:
    router: str        # "custom" | "routellm" | "notdiamond" | "difficulty_baseline"
    strength: str      # STRONG | CHEAP
    grounded: bool     # True only for a router that can express the grounding decision
    raw_model: str     # the router's actual pick / label, kept for the record
