"""
Shared citation-extraction utility.

Used by every client in clients/ (foundation, instruction-tuned, SLM) to
pull a Florida Building Code / Naples amendment section number out of raw model text,
so it can be compared against `gold_section` in data/fbc_eval_questions.csv.

This is deliberately a regex extractor, not an LLM-based one — the whole point of the
citation-match metric is a deterministic, non-judged signal that sits alongside the
LLM-as-judge score. If the extractor itself used an LLM, you'd be measuring the judge's
reliability, not the model's.
"""

import re

# Matches patterns like:
#   FBC 1006.2.1        FBC Section 1020.2       Florida Building Code 1620.1
#   Chapter 10, Section 1006.2                    R301.2.1.2
#   Naples Code Sec. 58-1(a)                      Sec. 4.02.03 (Collier County LDC style)
_SECTION_PATTERNS = [
    # "FBC 1006.2.1" / "FBC Section 1020.2" / "Florida Building Code 1620.1"
    # / "FBC Building 1020.2" / "Florida Building Code, Residential, 322.1"
    # (optional volume word between the code name and the section number)
    r"(?:FBC|Florida Building Code),?\s*"
    r"(?:Building|Residential|Existing Building|Energy Conservation|Mechanical|Plumbing|Fuel Gas|Accessibility)?,?\s*"
    r"(?:Section|Sec\.?|Ch(?:apter)?\.?\s*\d+,?\s*Section)?\s*(\d{3,4}(?:\.\d+)*)",
    # Residential-code style prefixes: R301.2.1.2, N1101.1
    r"\b([RN]\d{3,4}(?:\.\d+)*)\b",
    # Naples / Collier County municipal code style: "Sec. 58-1071(a)" or "Section 4.02.03"
    r"(?:Sec(?:tion)?\.?)\s*(\d+[-.]\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?)",
    # County/municipal ordinances: "Ordinance 2023-64" (first ordinance cited wins)
    r"\bOrdinance\s+(\d{4}-\d+)\b",
    # Collier Land Development Code style: "LDC 4.02.01" / "LDC Section 3.04.02"
    r"\bLDC\s*(?:Section|Sec\.?)?\s*(\d+(?:\.\d+)+)",
    # Bare "Chapter 16" references with no finer section — lowest-confidence fallback
    r"\bChapter\s+(\d{1,2})\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _SECTION_PATTERNS]


def extract_section_citation(text: str) -> str | None:
    """
    Return the first plausible FBC/Naples section citation found in `text`, or None if
    no citation-shaped substring is present.

    This does not verify the section actually exists or is correct — that check
    happens downstream in the notebook (Section 2/6), by comparing against
    `gold_section` and, for the grounded variant, against the section numbers actually
    present in the supplied context. This function only answers: "did the model claim
    a specific section at all, and what did it claim?"
    """
    if not text:
        return None

    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            return _normalize(match.group(1))

    return None


def _normalize(raw_section: str) -> str:
    """
    Normalize whitespace/case so that '1006.2.1', '1006.2.1 ', and '1006.2.1.' all
    compare equal to a gold value of '1006.2.1'.
    """
    return raw_section.strip().rstrip(".").upper()


def normalize_section(raw_section: str | None) -> str | None:
    """
    Normalize an ALREADY-ISOLATED section value (e.g. the `section` field of a
    structured JSON answer, which is a bare "1003.2" with no prose around it).

    Unlike extract_section_citation(), this does NOT require a "FBC"/"Section"
    prefix — the caller has already told us this string *is* the citation, so a
    bare number is valid here. Use extract_section_citation() for free prose and
    this for a dedicated citation field:

        cited = extract_section_citation(field) or normalize_section(field)

    Returns None for empty/whitespace input so "no section" stays distinguishable.
    """
    if not raw_section or not raw_section.strip():
        return None
    return _normalize(raw_section)


def section_present_in_context(section: str | None, context: str) -> bool:
    """
    Used in Lesson 4's grounding-compliance check: does the section the model cited
    actually appear in the retrieved context passage, or did it invent one that merely
    looks plausible? A citation can pass extract_section_citation() but still fail this
    check if the model hallucinated a real-looking section number not present in the
    text it was given.
    """
    if not section:
        return False
    return section.lower() in context.lower()


if __name__ == "__main__":
    # Quick manual sanity checks — run with `python clients/citation_utils.py`
    examples = [
        "Per FBC Section 1020.2, the minimum corridor width is 44 inches.",
        "This is governed by FBC 1006.2.1 and its exceptions.",
        "See Chapter 16 of the Florida Building Code for wind load requirements.",
        "Naples requires this under Sec. 58-1071(a) of the municipal code.",
        "R301.2.1.2 covers wind-borne debris regions.",
        "I don't have enough information to answer this question.",
    ]
    for ex in examples:
        print(f"{extract_section_citation(ex)!r:15} <- {ex}")