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

**Phase 2 — Generation:** complete.

- Unified multi-provider LLM client implemented in [src/clients/unified_llm.py](src/clients/unified_llm.py) with SQLite-backed caching at `data/llm_cache.sqlite`. Wraps OpenAI, Anthropic, DeepSeek, and Groq behind one interface so identical calls are billed once.
- Centralized model + path config in [src/config.py](src/config.py) — single source of truth for model IDs, generation parameters, and pipeline paths.
- Generation driver in [src/generate.py](src/generate.py) — loops over all 120 prompts × 4 providers (480 responses), saves each to `data/responses/{provider}/{prompt_id}.json`, and is restart-safe (skips already-saved files; cache hits are free).
- **Provider lineup change:** Gemini was replaced with DeepSeek mid-run because Gemini's free-tier quota and `gemini-1.5-flash` v1beta deprecation made full coverage unreliable. The 20 partial Gemini responses were discarded; all 120 DeepSeek responses generated cleanly.
- **Final response counts** under [data/responses/](data/responses/):

| Provider  | Saved | Expected | Status   |
|-----------|-------|----------|----------|
| OpenAI    | 120   | 120      | Complete |
| Anthropic | 120   | 120      | Complete |
| DeepSeek  | 120   | 120      | Complete |
| Groq      | 120   | 120      | Complete |
| **Total** | **480** | **480** | **100%** |

**Models used (April 2026)**

| Provider  | Model ID                          | Display name      |
|-----------|-----------------------------------|-------------------|
| OpenAI    | `gpt-4o-mini`                     | GPT-4o-mini       |
| Anthropic | `claude-haiku-4-5-20251001`       | Claude Haiku 4.5  |
| DeepSeek  | `deepseek-v4-flash`               | DeepSeek V4 Flash |
| Groq      | `llama-3.3-70b-versatile`         | Llama 3.3 70B     |

Generation params: `temperature=0.7`, `max_tokens=1500` (uniform across providers).

**Phase 3 — Extraction:** complete (480 / 480, validated).

- Extraction driver in [src/extract.py](src/extract.py) reads each free-text response from [data/responses/](data/responses/) and uses **GPT-4o-mini** (`temperature=0`, `max_tokens=8000`) to convert it into a strict structured-JSON object saved at `data/extractions/{provider}/{prompt_id}.json`.
- Extraction prompt template lives in [prompts/extraction.txt](prompts/extraction.txt) — the schema covers all three response shapes the benchmark elicits (nutrition, fitness, lifestyle/wellness) and exposes ten top-level blocks the downstream judges consume:
  - `summary`, `response_type`, `primary_goal`
  - `meal_components`, `all_ingredients`, `all_dishes_or_foods_named`
  - `fitness_components`, `routine_structure`
  - `cost_information` (budget tier, aid programs, store types)
  - `cultural_signals` (cuisines, religious/spiritual frameworks, fasting observances, celebrations)
  - `feasibility_signals` (kitchen access, equipment, environmental constraints)
  - `household_and_demographic_context`
  - `medical_or_health_signals`
  - `constraint_adherence` (DAGMetric ground truth)
  - `caveats_and_disclaimers`, `extraction_notes`
- Validator [src/validate_extractions.py](src/validate_extractions.py) confirms each file is valid JSON and flags suspicious classifications (e.g., `response_type='fitness_plan'` with empty component arrays).
- **Token-limit lessons:** the extraction `max_tokens` was raised twice — 2500 → 4000 → 8000 — because the dense schema can produce 14k+ characters of JSON for full 7-day plans. Final pass produced 478 clean files and 2 suspicious-but-valid Anthropic outputs (advisory-style responses with no concrete plan items, manually verified).
- **Cost note:** because the unified client caches by `(provider, model, prompt, params)`, every parameter bump invalidates the cache and forces fresh API calls — keep this in mind before changing extraction parameters.

**Repository scaffold:** complete. Source files for grounding, judges, similarity, ML baseline, aggregation, and validation are stubbed in [src/](src/) for the upcoming phases.

## Project structure

```
accessible-health-bench/
├── data/
│   ├── LLM_Prompts.csv               # source spreadsheet (120 rows)
│   ├── prompts.jsonl                 # converted, validated prompts
│   ├── responses/                    # raw model responses (480, gitignored)
│   ├── extractions/                  # structured JSON from responses (480, gitignored)
│   ├── enriched/                     # extractions + Wikidata tags (gitignored)
│   └── llm_cache.sqlite              # cached API calls (gitignored)
├── src/
│   ├── config.py                     # model IDs, paths, generation params
│   ├── clients/unified_llm.py        # multi-provider LLM client w/ caching
│   ├── generate.py                   # Phase 2 driver — 480 responses
│   ├── extract.py                    # Phase 3 — structured extraction
│   ├── validate_extractions.py       # Phase 3 validator (parse + heuristics)
│   ├── grounding/                    # Wikidata SPARQL queries + cache
│   ├── judges/                       # G-Eval / DAGMetric judges
│   ├── arena_eval.py                 # ArenaGEval pairwise matrix
│   ├── similarity.py                 # Sentence-BERT cosine similarity
│   ├── ml_baseline.py                # logistic regression baseline
│   └── aggregate.py                  # combine scores → results CSVs
├── prompts/                          # version-controlled judge prompt templates
│   └── extraction.txt                # Phase 3 extraction template (10-block schema)
├── scripts/csv_to_jsonl.py           # CSV → JSONL converter
├── notebooks/                        # 01_eda, 02_results, 03_figures
├── dashboard/                        # React + Vite results viewer
├── results/                          # scores, kappa, arena matrix, figures
└── paper/main.tex
```

## Roadmap

- **Phase 4 — Grounding:** Wikidata enrichment of extracted records (cultural food tags, affordability anchors) → `data/enriched/`.
- **Phase 5 — Judging:** four G-Eval / DAGMetric judges (affordability, cultural, adherence, feasibility) + ArenaGEval pairwise matrix.
- **Phase 6 — Analysis:** aggregation, Cohen's kappa validation, figures, dashboard, paper.

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
   Add values for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`. **Never commit `.env`.**

## Common commands

Regenerate the prompt set from CSV:
```
python scripts/csv_to_jsonl.py
```

Run a small generation pilot (5 prompts × 4 providers = 20 calls):
```
python -m src.generate --pilot 5
```

Run the full generation pass (120 prompts × 4 providers = 480 calls; cached calls are free):
```
python -m src.generate
```

Run extraction on every saved response (skips files already extracted):
```
python -m src.extract
```

Run extraction in pilot mode (first N responses per provider):
```
python -m src.extract --pilot 10
```

Validate every extracted file (parse-error + suspicious-content heuristics):
```
python -m src.validate_extractions
```

Smoke-test the unified client across all 4 providers (~$0.05):
```
python -m src.clients.unified_llm
```
