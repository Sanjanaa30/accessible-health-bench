# Accessible Health Bench

A benchmark for evaluating large language models on accessible health-information tasks across four dimensions — affordability, cultural appropriateness, adherence to stated constraints, and feasibility — with Wikidata-grounded enrichment and LLM-as-judge scoring. The benchmark probes whether models maintain quality and constraint-following when prompts include real-world accessibility constraints (limited budget, cultural food norms, lifestyle limitations).

## Status

**Phase 1 — Dataset Construction:** complete.

- 120 prompts authored across 3 categories (financial, cultural, lifestyle), 60 base prompts × 2 variants (baseline / constrained). See [data/prompts.jsonl](data/prompts.jsonl).
- Source spreadsheet: [data/LLM_Prompts.csv](data/LLM_Prompts.csv).
- Conversion script: [scripts/csv_to_jsonl.py](scripts/csv_to_jsonl.py) — emits one JSON object per prompt with fields `id`, `category`, `variant`, `prompt_text`, `category_type`, and `stated_constraints` (the structured ground truth consumed by the DAGMetric judge).

| Category   | Baseline | Constrained | Total |
|------------|----------|-------------|-------|
| Financial  | 20       | 20          | 40    |
| Cultural   | 20       | 20          | 40    |
| Lifestyle  | 20       | 20          | 40    |
| **Total**  | **60**   | **60**      | **120** |

**Phase 2 — Generation:** mostly complete (3 of 4 providers fully done).

- Unified multi-provider LLM client implemented in [src/clients/unified_llm.py](src/clients/unified_llm.py) with SQLite-backed caching at `data/llm_cache.sqlite`. Wraps OpenAI, Anthropic, Google Gemini, and Groq behind one interface so identical calls are billed once.
- Centralized model + path config in [src/config.py](src/config.py) — single source of truth for model IDs, generation parameters, and pipeline paths.
- Generation driver in [src/generate.py](src/generate.py) — loops over all 120 prompts × 4 providers (480 responses), saves each to `data/responses/{provider}/{prompt_id}.json`, and is restart-safe (skips already-saved files; cache hits are free).
- **Current response counts** under [data/responses/](data/responses/):

| Provider  | Saved | Expected | Status      |
|-----------|-------|----------|-------------|
| OpenAI    | 120   | 120      | Complete    |
| Anthropic | 120   | 120      | Complete    |
| Groq      | 120   | 120      | Complete    |
| Gemini    | 20    | 120      | In progress |
| **Total** | **380** | **480** | 79%        |

Remaining 100 Gemini responses pending (re-run `python -m src.generate` — restart-safe, will skip the 380 already saved).

**Models used (April 2026)**

| Provider  | Model ID                          | Display name      |
|-----------|-----------------------------------|-------------------|
| OpenAI    | `gpt-4o-mini`                     | GPT-4o-mini       |
| Anthropic | `claude-haiku-4-5-20251001`       | Claude Haiku 4.5  |
| Gemini    | `gemini-2.5-flash`                | Gemini 2.5 Flash  |
| Groq      | `llama-3.3-70b-versatile`         | Llama 3.3 70B     |

Generation params: `temperature=0.7`, `max_tokens=1500` (uniform across providers).

**Repository scaffold:** complete. Source files for grounding, judges, similarity, ML baseline, aggregation, and validation are stubbed in [src/](src/) for the upcoming phases.

## Project structure

```
accessible-health-bench/
├── data/
│   ├── LLM_Prompts.csv         # source spreadsheet (120 rows)
│   ├── prompts.jsonl           # converted, validated prompts
│   ├── responses/              # raw model responses (gitignored)
│   ├── extractions/            # structured JSON from responses
│   ├── enriched/               # extractions + Wikidata tags
│   └── llm_cache.sqlite        # cached API calls (gitignored)
├── src/
│   ├── config.py               # model IDs, paths, generation params
│   ├── clients/unified_llm.py  # multi-provider LLM client w/ caching
│   ├── generate.py             # Phase 2 driver — 480 responses
│   ├── extract.py              # Phase 3 — structured extraction
│   ├── grounding/              # Wikidata SPARQL queries + cache
│   ├── judges/                 # G-Eval / DAGMetric judges
│   ├── arena_eval.py           # ArenaGEval pairwise matrix
│   ├── similarity.py           # Sentence-BERT cosine similarity
│   ├── ml_baseline.py          # logistic regression baseline
│   ├── aggregate.py            # combine scores → results CSVs
│   └── validate.py             # Cohen's kappa inter-rater reliability
├── prompts/                    # version-controlled judge prompt templates
├── scripts/csv_to_jsonl.py     # CSV → JSONL converter
├── notebooks/                  # 01_eda, 02_results, 03_figures
├── dashboard/                  # React + Vite results viewer
├── results/                    # scores, kappa, arena matrix, figures
└── paper/main.tex
```

## Roadmap

- **Phase 2 (current):** finish the remaining 100 Gemini responses to reach the full 480.
- **Phase 3 — Extraction & Grounding:** structured JSON extraction into [data/extractions/](data/extractions/) + Wikidata enrichment into [data/enriched/](data/enriched/) (cultural food tags, affordability anchors).
- **Phase 4 — Judging:** four G-Eval / DAGMetric judges (affordability, cultural, adherence, feasibility) + ArenaGEval pairwise matrix.
- **Phase 5 — Analysis:** aggregation, Cohen's kappa validation, figures, dashboard, paper.

## Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1     # PowerShell
   # or:  source .venv/Scripts/activate    # Git Bash on Windows
   # or:  source .venv/bin/activate        # macOS / Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy and fill in your API keys:
   ```
   cp .env.example .env
   ```
   Add values for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`. **Never commit `.env`.**

## Common commands

Regenerate the prompt set from CSV:
```
python scripts/csv_to_jsonl.py
```

Run a small pilot (5 prompts × 4 providers = 20 calls):
```
python -m src.generate --pilot 5
```

Run the full generation pass (120 prompts × 4 providers = 480 calls; cached calls are free):
```
python -m src.generate
```

Smoke-test the unified client across all 4 providers (~$0.05):
```
python -m src.clients.unified_llm
```
