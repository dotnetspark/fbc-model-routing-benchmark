# Model Selection Tutorial Series — Florida Building Code Assistant

**Phase 1 of the 8-Week AI Systems Architecture Learning Plan**

A tutorial series that teaches model-type selection by building a working assistant over the Florida Building Code (FBC) and the Naples, FL municipal code — a domain with real long-term utility (permitting, code compliance, contractor/AEC workflows) and a publicly available, citable corpus.

Each lesson routes the same class of question — "what does the code require here?" — through a foundation model, an instruction-tuned model, and a small language model (SLM), and measures latency, cost, and accuracy against the actual code text. You are not benchmarking abstract prompts; you are benchmarking whether each model class can be trusted to answer a compliance question correctly.

## Why this domain

- The Florida Building Code and local amendments (Naples / Collier County) are public documents with clear, checkable answers — ideal for building a deterministic evaluation set instead of relying on subjective judging alone.
- Code text is dense, cross-referenced, and full of exceptions — a good stress test for grounding, citation accuracy, and hallucination detection.
- The end state (a groundable, citation-accurate code-lookup assistant) is a real, reusable tool, not a throwaway demo.

## Structure

```
model-selection-tutorial/
├── README.md                          ← you are here
├── project.md                         ← architecture, scope, milestones, deliverables
├── .gitignore
├── LICENSE
├── GITHUB_ABOUT.md
├── lessons/
│   ├── 01-foundation-llms.md
│   ├── 02-instruction-tuned-models.md
│   ├── 03-small-language-models.md
│   ├── 04-multimodal-models.md
│   ├── 05-fine-tuned-models.md
│   ├── 06-rag-integrated-models.md
│   └── 07-when-to-use-each-model-type.md
├── clients/
│   ├── foundation_client.py
│   ├── instruction_tuned_client.py
│   ├── slm_client.py
│   └── multimodal_client.py
├── engine/
│   ├── finetune_breakeven.py
│   └── router.py
├── data/
│   └── fbc_eval_questions.csv
├── results/
│   └── *.jsonl
└── analysis/
    └── benchmark_report.ipynb
```

The notebook is intentionally separate from the harness code: `clients/`, `engine/`, and the FastAPI orchestrator (built across the lessons) stay as plain importable Python modules so the benchmark is runnable headlessly (`python run_benchmark.py`) and CI-friendly. The notebook consumes their output — it never re-implements inference logic — so there's exactly one source of truth for how a number was produced.

## How to use this material

Each lesson follows the same structure:

1. **Concept** — the technical substance, with citations to primary sources
2. **Why it matters for routing decisions**
3. **Build increment** — the piece of the assistant you implement after reading
4. **Checkpoint** — how you know the increment works, against real code text

Work through lessons in order. By Lesson 7 you will have a working, citation-grounded Florida Building Code assistant with a decision engine that recommends which model class to use for a given query type, backed by measured data.

## Source material

- Florida Building Code (current edition), published by the Florida Building Commission — obtain the public ICC-hosted or Florida DBPR-linked version
- City of Naples / Collier County local amendments and municipal code (public records)

Use only the officially published text as your ground-truth corpus. Do not rely on model memory for code citations — that is precisely the failure mode this series is designed to catch.

## Prerequisites

- Python 3.11+
- API access to at least one hosted foundation/instruction-tuned model (OpenAI, Azure OpenAI, or Anthropic)
- Ollama or equivalent for local SLM inference (Phi-3, Llama 3.2 3B, or Qwen2.5 3B)
- A local copy of the relevant FBC chapters and Naples municipal code sections (PDF or text) for the evaluation set and later RAG grounding

## License

MIT — fork it, point it at your own jurisdiction's code, publish your own numbers.
