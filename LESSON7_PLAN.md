# Lesson 7 Plan — Routing-behavior comparison: custom vs. RouteLLM vs. NotDiamond

**Self-contained execution plan.** Written so a fresh agent (Copilot CLI) or a human can finish Lesson 7 without prior chat context. Follow the phases in order; each has a verification step. Commit as you go.

> **Execution status (updated during the build — corrections to the sketches below):**
> - **Env:** built in a **separate `.venv311`** (not by replacing `.venv`, which was file-locked by the IDE). Python 3.11.13 via `uv`. Everything works; the 3.13 `.venv` is untouched.
> - **RouteLLM: runs in Docker** (not on the Windows host — its `litellm→torch→transformers→datasets` stack has no working Windows install path). [`docker/Dockerfile`](docker/Dockerfile) + [`docker-compose.yml`](docker-compose.yml) build a `python:3.11-slim` image; `docker compose run --rm routers` executes the dry-run and writes `results/router_selections.csv` back to the host. Uses RouteLLM's **local `bert` router** (the `mf` router embeds each prompt via the OpenAI API — a real network call; `bert` stays offline). `routers/difficulty_baseline.py` is kept as a transparent corroborating heuristic. `run_router_dryrun.py` imports `routers/routellm_router.py` only when `routellm` is importable, so the host driver still runs without it.
> - **NotDiamond API (real, verified end-to-end):** env var is **`NOTDIAMOND_API_KEY`** (no middle underscore — confirmed against the docs; an earlier note here said `NOT_DIAMOND_API_KEY`, which was wrong). The router passes the key explicitly and accepts either spelling. Call is `client.model_router.select_model(llm_providers=[{"provider","model"}], messages=[...], tradeoff="cost")`; the pick is `resp.providers[0]` (`.provider`, `.model`). Candidate model strings must be in NotDiamond's catalog (many 400) — verified set: `gpt-4o` / `claude-3-7-sonnet-latest` (strong), `gpt-4o-mini` / `claude-3-haiku-20240307` (cheap). Result: 42% strong / 58% cheap, 0% grounded. NotDiamond also supports training a custom router (`train_custom_router`) on eval data — extensible on the model axis, but still no grounding input.
> - **Done (all four routers real/working):** `routers/` (base, custom, notdiamond_router, difficulty_baseline, routellm_router), the Docker setup, `run_router_dryrun.py`, `analysis/router_comparison.py`, the selection-gravity chart (custom + RouteLLM + NotDiamond + illustrative heuristic), FINDINGS § Lesson 7 + Design/parameter-provenance, lessons/07 checkpoint. **Regenerate:** `docker compose run --rm routers` (needs `NOTDIAMOND_API_KEY` in `.env`) then `.venv311/Scripts/python analysis/router_comparison.py`. **Remaining:** merge `feat/lesson7_router`.

---

## 0. Context (read first)

This repo benchmarks three model tiers on Florida Building Code / Naples-Collier local-code citation, cold vs. grounded. **Lessons 1, 2, 3, 6 are done and merged to `main`.** The synthesis is in [`FINDINGS.md`](FINDINGS.md); per-lesson detail is in [`lessons/`](lessons/); the analysis notebook is [`analysis/benchmark_report.ipynb`](analysis/benchmark_report.ipynb); raw results are in [`results/`](results/).

**Key measured numbers (from FINDINGS.md — the custom router encodes these):**

| tier | cold correct | grounded correct | jurisdiction cold→grounded |
|---|---|---|---|
| Opus 4.8 (strong) | 25.8% | 59.1% | 28.8% → 63.6% |
| Haiku 4.5 (cheap, schema) | 29.5% | 77.3% | ~ → 81.8% |
| phi3:mini (cheap, local) | 7.9% | 52.5% | 10.8% → 70.0% |

**Lesson 7 goal.** *Dry-run* three routers over the 45 evaluation questions — record **which model each router selects**, not the answer (near-zero cost: routing logic runs without calling the target LLM). Plot the selection "gravity" on two axes — **model strength** (strong↔cheap) and **grounded? (yes/no)** — and give a data-referenced recommendation.

**The thesis (what this proves).** Generic routers (RouteLLM, NotDiamond) optimize *prompt difficulty → model tier*; their ranking is embedded/closed and **cannot express the grounding decision**. Our measurements show grounding is the *dominant* lever (grounded-cheap beats cold-strong). So on the strength×grounding plot, **the entire "grounded" band is empty for every router but the custom one** — the extensibility gap made visible. The recommendation follows from the independently-measured Lessons 1–6 data, not from a self-scored contest.

**Honesty guardrails (do not skip — they make the result credible):**
- This is a **descriptive study of selection behavior**, not an accuracy bake-off. Do NOT claim "custom router wins on held-out accuracy" — the custom router is built *from* the findings, so that would be circular. The claim is about **expressiveness** (generic routers can't route on grounding) plus **alignment with measured performance**.
- Small n (45 questions; some categories n=3–4). Label category-level results "illustrative."
- Routers use **different model catalogs** (RouteLLM: a strong/weak pair; NotDiamond: its own list). Do NOT compare exact model names — interpret one level up, at **strength tier + grounding axis**.

---

## 1. Environment — Python 3.11 (RouteLLM requires it; current venv is 3.13)

RouteLLM needs Python 3.11. NotDiamond supports 3.9–3.12. The existing `.venv` is 3.13 → create a fresh **3.11** venv and make it the project environment.

**Recommended path — `uv` (cleanest multi-Python manager on Windows):**
```powershell
# install uv if absent: https://docs.astral.sh/uv/  (winget install astral-sh.uv)
uv python install 3.11
# from repo root:
uv venv .venv --python 3.11          # replaces .venv with a 3.11 one (delete old first if needed)
.\.venv\Scripts\activate
```

**Fallback — official installer:**
```powershell
winget install Python.Python.3.11          # or download python-3.11.x-amd64.exe from python.org
py -3.11 -m venv .venv
.\.venv\Scripts\activate
```

**Reinstall project deps + routers into the 3.11 venv:**
```powershell
pip install anthropic google-genai python-dotenv pandas matplotlib jupyter ipykernel pymupdf pydantic
pip install routellm            # per https://github.com/lm-sys/RouteLLM ; may pull torch — see gotchas
pip install notdiamond          # per https://docs.notdiamond.ai
python -m ipykernel install --user --name fbc-311 --display-name "Python 3.11 (fbc)"
```

**Verify (all must pass before continuing):**
```powershell
python --version                 # 3.11.x
python -c "import routellm; import notdiamond; print('routers OK')"
python -c "import run_benchmark; print('harness still imports')"   # 3.11 satisfies the 3.10+ code
```
If RouteLLM fails to install/import on Windows (torch/C-deps), see **Gotchas → RouteLLM fallback** and substitute the transparent difficulty baseline; the study still stands.

**Point the notebook at the 3.11 kernel** ("Python 3.11 (fbc)") when re-executing later. Re-run `jupyter nbconvert --to notebook --execute --inplace analysis/benchmark_report.ipynb` once to confirm the existing report still runs on 3.11 (expect 0 errors, 8 charts).

---

## 2. Secrets (`.env`, never commit)

Add:
```
NOTDIAMOND_API_KEY=...        # from https://app.notdiamond.ai (free tier exists)
```
- NotDiamond `model_select` returns a recommendation **without running inference** → dry-run, minimal/no cost.
- RouteLLM's `mf` router runs a local classifier (downloads a checkpoint from HuggingFace on first use); routing decisions need **no** OpenAI key. Only pass `strong_model`/`weak_model` as *labels* — we never call them.

---

## 3. Canonical router interface — `routers/base.py`

```python
from dataclasses import dataclass

@dataclass
class RouterChoice:
    router: str            # "custom" | "routellm" | "notdiamond"
    strength: str          # "strong" | "cheap"   (interpret every router's pick to this)
    grounded: bool         # True only a router that can decide to ground can set this
    raw_model: str         # the router's actual pick, for the record

class Router:              # each concrete router subclasses / duck-types this
    name: str
    def select(self, question: str, category: str) -> RouterChoice: ...
```

Strength mapping is the leveling device: every router's concrete pick collapses to `strong` or `cheap`. Only the **custom** router ever sets `grounded=True`.

---

## 4. Custom router — `routers/custom.py`

Encodes the measured findings: **ground everything groundable, then pick the cheapest tier that clears an accuracy floor.**

```python
# Measured grounded correct-rate by category (from FINDINGS / results). Cheap = Haiku(schema)/phi3.
# Rule of thumb per the data: grounded Haiku (cheap) clears a useful floor on every groundable
# category, so custom routes cheap+grounded for those; only "definitional" (tiny n, low grounded
# accuracy) or a non-groundable item falls back to strong.
GROUNDABLE = {"jurisdiction_amendment", "numeric", "state_amendment"}
FLOOR = 0.50   # minimum acceptable grounded correct-rate to keep it on the cheap tier

# grounded cheap-tier accuracy by category (Haiku-schema, from measured runs — update from results):
GROUNDED_CHEAP_ACC = {
    "jurisdiction_amendment": 0.82, "numeric": 0.75, "state_amendment": 0.67, "definitional": 0.67,
}

def select(question, category):
    grounded = category in GROUNDABLE or category == "definitional"
    cheap_ok = grounded and GROUNDED_CHEAP_ACC.get(category, 0) >= FLOOR
    strength = "cheap" if cheap_ok else "strong"
    return RouterChoice("custom", strength, grounded, raw_model=f"{strength}{'+grounded' if grounded else ''}")
```
- Keep the accuracy numbers in a small table sourced from `results/` (or hardcode from FINDINGS with a comment pointing at the source). The point is transparency: the rule is a readable table you can extend — the opposite of an embedded router.

---

## 5. RouteLLM router — `routers/routellm_router.py`

Dry-run = get the strong/weak routing decision per prompt **without** calling any LLM. RouteLLM's `mf` router outputs a win-probability; compare to a calibrated threshold.

```python
# VERIFY method names against the current README: https://github.com/lm-sys/RouteLLM
from routellm.controller import Controller
import pandas as pd

_ctrl = Controller(routers=["mf"],
                   strong_model="gpt-4-1106-preview",     # labels only; never called
                   weak_model="mixtral-8x7b-instruct-v0.1")

# Win-rate = P(strong meaningfully better). Threshold: use RouteLLM's calibrate_threshold for a
# target strong-call fraction (e.g. 50%) so the baseline isn't rigged, OR document a fixed 0.5.
_THRESHOLD = 0.5   # replace with _ctrl.calibrate_threshold(...) if available; document the choice

def select(question, category):
    wr = _ctrl.batch_calculate_win_rate(pd.Series([question]), router="mf").iloc[0]   # VERIFY API
    strength = "strong" if wr >= _THRESHOLD else "cheap"
    return RouterChoice("routellm", strength, grounded=False, raw_model=f"mf/{'strong' if strength=='strong' else 'weak'}")
```
- `grounded=False` **always** — RouteLLM has no grounding concept. That is the finding, not a limitation to fix.
- If `batch_calculate_win_rate` isn't the current API, use whatever the README exposes to obtain a per-prompt routing decision without inference (e.g. the router object's scoring method). Document the exact call and threshold you used.

---

## 6. NotDiamond router — `routers/notdiamond_router.py`

`model_select` returns the recommended model **without** running it → dry-run.

```python
# VERIFY against https://docs.notdiamond.ai — SDK surface may differ by version.
from notdiamond import NotDiamond

# Candidate set: one strong + one cheap from NotDiamond's supported catalog (NOT our exact tiers).
_STRONG = {"openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"}
_CHEAP  = {"openai/gpt-4o-mini", "anthropic/claude-3-5-haiku-20241022"}
_client = NotDiamond(llm_configs=list(_STRONG | _CHEAP))   # reads NOTDIAMOND_API_KEY from env

def select(question, category):
    result, session_id, provider = _client.model_select(          # VERIFY: may be client.chat.completions.model_select
        messages=[{"role": "user", "content": question}],
        tradeoff="cost",                                          # optional; document what you pass
    )
    picked = f"{provider.provider}/{provider.model}"              # VERIFY attribute names
    strength = "strong" if picked in _STRONG else "cheap"
    return RouterChoice("notdiamond", strength, grounded=False, raw_model=picked)
```
- `grounded=False` always — NotDiamond routes model only.
- If ND's catalog lacks these exact ids, pick its nearest strong/cheap and update the sets. The mapping is by tier, not exact model.

---

## 7. Dry-run driver — `run_router_dryrun.py`

```python
import csv
from pathlib import Path
from dotenv import load_dotenv
from routers import custom, routellm_router, notdiamond_router   # each exposes select()

load_dotenv()
ROUTERS = {"custom": custom.select, "routellm": routellm_router.select, "notdiamond": notdiamond_router.select}
rows = list(csv.DictReader(open("data/fbc_eval_questions.csv", encoding="utf-8")))
out = []
for r in rows:
    for name, fn in ROUTERS.items():
        try:
            c = fn(r["question"], r["category"])
            out.append({"question_id": r["question_id"], "category": r["category"],
                        "router": name, "strength": c.strength,
                        "grounded": c.grounded, "raw_model": c.raw_model})
        except Exception as e:
            out.append({"question_id": r["question_id"], "category": r["category"],
                        "router": name, "strength": "ERROR", "grounded": None, "raw_model": str(e)[:80]})
with open("results/router_selections.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["question_id","category","router","strength","grounded","raw_model"])
    w.writeheader(); w.writerows(out)
print(f"wrote {len(out)} selections -> results/router_selections.csv")
```
Run it; expect 45×3 = 135 rows, ERROR count 0 (or investigate). **No target-LLM inference** — only NotDiamond's/RouteLLM's internal routing.

---

## 8. Analysis + charts (add a Section 9 to the notebook, or `analysis/router_comparison.py`)

Reuse the notebook's palette (`CATEGORICAL`, `STATUS`, `INK/MUTED`, rcParams already defined in the imports cell).

- **Chart A — selection gravity (the headline).** For each router, show the distribution of picks over the 2×2 grid `strength ∈ {cheap,strong}` × `grounded ∈ {no,yes}`. Small-multiples (one 2×2 heatmap/bar per router), cell value = count of the 45 questions. **The visual payoff: the entire `grounded=yes` band is empty for `routellm` and `notdiamond`, populated only for `custom`.** Call that out in the caption.
- **Chart B — routing mix by query category.** Per router, a stacked bar over categories showing the strength+grounding mix. Shows generic routers can only vary the strength within a category; custom varies grounding too.
- **Overlay with measured performance.** Annotate each router's dominant pick with what it *achieves* per Lessons 1–6 (e.g. "RouteLLM → strong+ungrounded → 26% measured correct"; "custom → cheap+grounded → 52–82% measured"). Pull these from `results/` or the FINDINGS table so the judgment is data-referenced, not asserted.

Re-execute the notebook (0 errors expected). Extract the charts to PNG and eyeball for label/geometry issues (the notebook already has an extraction pattern in scratch usage; follow the dataviz "render it and look at it" step).

---

## 9. Recommendation write-up

Add a **"Lesson 7 — routing behavior"** section to `FINDINGS.md` and fill `lessons/07-when-to-use-each-model-type.md` § Checkpoint with the measured story:
- Generic routers optimize model-tier on difficulty; the ranking is embedded/closed; they **structurally cannot route on grounding** (show the empty grounded band).
- Measured data shows grounding is the dominant lever, so a difficulty router that upgrades hard FBC questions to the strong model *without grounding* optimizes the wrong axis and still lands ~26%.
- **Recommendation:** for verifiable-answer domains, put a thin, extensible **grounding-first** decision in front of routing (even a lookup table beats a closed tier-router); use an off-the-shelf router, if at all, only for the *tier* choice *after* the grounding decision.
- Restate the honesty guardrails from §0 (descriptive study; custom encodes findings; small n; strength+grounding leveling).

Update `README.md` structure/prereqs to mention the 3.11 requirement for the router work, and `project.md` M7 as done.

---

## 10. Deliverables + commit plan (branch `feat/lesson7_router`, then `--ff-only` merge to `main`)

Commit 1 — infra:
- `routers/` (`base.py`, `custom.py`, `routellm_router.py`, `notdiamond_router.py`), `run_router_dryrun.py`
- `.gitignore`: any RouteLLM/HF model cache dirs if they land in-repo
- message: `feat: Lesson 7 router dry-run harness (custom / RouteLLM / NotDiamond)`

Commit 2 — results + analysis:
- `results/router_selections.csv`, notebook Section 9 (or `analysis/router_comparison.py`), `FINDINGS.md` + `lessons/07` + `README`/`project.md` updates
- message: `feat: Lesson 7 results — routing behavior comparison + recommendation`

End `git commit` messages with:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
(or your own attribution if executed by another agent).

---

## 11. Gotchas & fallbacks

- **RouteLLM on Windows** may fail to install (torch / native deps). Fallback: implement `routers/difficulty_baseline.py` — a transparent difficulty→tier router (e.g. an LLM-judged 1–5 difficulty score, or a length/keyword heuristic) labeled `"difficulty-baseline"`, `grounded=False`. It makes the same point (tier-only, no grounding) and is fully reproducible. Swap it in for `routellm` in the driver if needed; note the substitution in the write-up.
- **NotDiamond SDK drift** — method/attribute names (`model_select`, `provider.model`, `llm_configs`) may differ by version. Verify against current docs; the *intent* is "given a prompt + candidate models, return the recommended model without inference."
- **Catalog mismatch** — neither RouteLLM nor NotDiamond routes among Opus 4.8 / Haiku 4.5 / phi3. Never map by exact model; map by **strength tier**. State this in the write-up.
- **Keys** stay in `.env`; never commit. NotDiamond free tier should cover 45 dry-run selects; confirm no per-call inference is billed (`model_select` is selection-only).
- **Circularity** is handled by the descriptive framing — keep it that way. If tempted to report a head-to-head accuracy number, use leave-one-out and label it illustrative, but the primary result is the selection-gravity chart + the expressiveness argument.
- **Notebook kernel** must be the 3.11 one after the env switch, or charts won't regenerate.
