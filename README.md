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

**Phase 4 — Grounding:** complete (480 / 480 enriched).

Each extraction is enriched against four external knowledge sources, producing a `grounding` block consumed by the Phase 5 judges plus a `_grounding_meta` block recording snapshot dates and SHA256s for paper-quality reproducibility.

| Source | Module | Purpose | Cache |
|---|---|---|---|
| Wikidata SPARQL + LLM fallback | [src/grounding/wikidata.py](src/grounding/wikidata.py) | Cuisine origin tags for ingredients/dishes (RQ2) | `data/wikidata_cache.sqlite` |
| BLS Average Retail Food Prices | [src/grounding/bls.py](src/grounding/bls.py) | Per-ingredient unit prices (RQ1) | `data/external/bls_prices.csv` |
| USDA Cost of Food at Home | [src/grounding/thrifty_plan.py](src/grounding/thrifty_plan.py) | Household-level weekly cost benchmarks (RQ1) | `data/external/usda_thrifty_plan.csv` |
| 2024 Adult Compendium of Physical Activities | [src/grounding/compendium.py](src/grounding/compendium.py) | MET values + WHO 2020 bucketing (RQ3) | `data/external/compendium_activities.csv` |

External reference data is built once via [src/download_external_data.py](src/download_external_data.py); the script writes a `data/external/MANIFEST.json` snapshot manifest so judges can record provenance.

**Three-pass orchestrator** in [src/ground_all.py](src/ground_all.py):
1. Ground each extraction; Wikidata accumulates misses across the entire run.
2. Single batched LLM-fallback call resolves all accumulated cuisine misses (one call instead of N).
3. Finalize and write all 480 enriched files to [data/enriched/](data/enriched/).

**Coverage outcomes (480 responses, after fallback):**

| Metric | Constrained (n=240) | Baseline (n=240) | Notes |
|---|---|---|---|
| Wikidata cuisine coverage | 52.4% | 60.1% | Constrained prompts introduce more region-specific ingredients that Wikidata can't always classify |
| BLS price coverage | 6.6% | 4.8% | Thin — BLS staple list (25 items) doesn't cover most international ingredients; affordability judge will lean on USDA Thrifty calibration |
| Compendium fitness coverage | 19.4% | 12.4% | Of fitness-bearing responses (89 constrained / 86 baseline) |
| Western centricity ratio | 3.8% | 4.2% | Constrained prompts produce slightly *less* Western content |
| Cost extractable | 13% | 0% | "Normalized" or "per_meal_extrapolated"; cultural/lifestyle prompts naturally have none |

`per_meal` budgets (e.g. "I have $5") are extrapolated to per-week using a 3-meals/day × 7-day = 21× multiplier and flagged as `per_meal_extrapolated` in [src/ground_all.py](src/ground_all.py) so the assumption is disclosable.

**Phase 4 reporter** [src/coverage_report.py](src/coverage_report.py) emits five paper-ready CSVs to [results/](results/) (split by `--variant constrained` or `--variant baseline`):
- `coverage_report.csv` — one row per response, all metrics
- `coverage_summary.csv` — provider × category aggregates (Table 1)
- `thrifty_classification.csv` — RQ1 affordability bucket distribution
- `feasibility_assessment.csv` — RQ3 WHO compliance distribution
- `cuisine_distribution.csv` — RQ2 top cuisines per provider × category

**Mini-pilot** [scripts/mini_pilot.py](scripts/mini_pilot.py) grounds exactly 6 prompts (one per sub-category) across all 4 providers — used to verify all three RQ pipelines fire end-to-end before scaling.

**Phase 5 — Evaluation:** in progress. Five tracks running in parallel:

| Track | Module | Status | Output |
|---|---|---|---|
| A. Four LLM judges (G-Eval + DAGMetric) | `src/judges/` + `src/run_judges.py` | TODO | `results/judge_scores.csv` |
| **B. Sentence-BERT similarity** | [src/similarity.py](src/similarity.py) | **complete** | [results/similarity.csv](results/similarity.csv) |
| C. Logistic regression baseline | [src/ml_baseline.py](src/ml_baseline.py) | gated on Phase 6 labels | `results/ml_baseline.csv` |
| D. ArenaGEval pairwise | [src/arena_eval.py](src/arena_eval.py) | TODO | `results/arena_matrix.csv` |
| E. Aggregation | [src/aggregate.py](src/aggregate.py) | TODO | `results/scores.csv` |

### Phase 5 Track B — Semantic adaptation (Sentence-BERT) — complete

Computes four distance signals between every (baseline, constrained) prompt-pair response per model. 60 prompt pairs × 4 providers = **240 paired comparisons** in [results/similarity.csv](results/similarity.csv).

**Four complementary signals** (per [src/similarity.py](src/similarity.py)):
1. `cosine_full` — Sentence-BERT (`all-MiniLM-L6-v2`) on full response text, chunked-and-mean-pooled for inputs > 1000 chars to avoid silent truncation.
2. `cosine_ingredients` — Sentence-BERT on the joined ingredient string.
3. `cosine_structural` — Sentence-BERT on a synthetic structural digest (response_type + meal_types + activity_types + kitchen access).
4. `jaccard_ingredients` — Set-based, length-invariant ingredient overlap. Uses the same `normalize_food_name` as Phase 4 grounding so `oats`/`oat` collapse to one set element.

**Headline result — model adaptation magnitude (cosine_full mean, n=20 per cell):**

| Provider | Cultural | Financial | Lifestyle |
|---|---|---|---|
| Anthropic Haiku 4.5 | **0.369** | **0.278** | 0.272 |
| Llama 3.3 70B (Groq) | 0.281 | 0.286 | 0.239 |
| DeepSeek V4 Flash | 0.282 | 0.205 | 0.193 |
| GPT-4o-mini | 0.270 | 0.206 | **0.154** |

Anthropic adapts the most across all three categories; OpenAI's GPT-4o-mini adapts the least. The gap between the most-adaptive cell (Anthropic cultural, 0.369) and the least-adaptive cell (OpenAI lifestyle, 0.154) is **2.4×**, consistent across signals.

**Pooled diagnostic:** mean cosine_full = 0.253, median = 0.224, between-cell σ = 0.054 (where each "cell" is one provider × category combination, n=20). Just below the soft "adaptive" threshold (mean ≥ 0.25, σ ≥ 0.06), but the per-cell spread is the real signal — uniform rigidity would have produced σ ≪ 0.04.

**Methodological finding — multi-signal design pays off.** Cosine and Jaccard tell different stories on financial prompts:
- `cosine_ingredients` (mean 0.21) suggests modest ingredient change.
- `jaccard_ingredients` (mean 0.80) reveals that ~80% of ingredients are actually replaced.

Cosine alone hides the swap because list lengths and surface phrasing stay similar. Jaccard catches what embedding similarity can't.

**Bimodality of Jaccard across all four models** (visible in `results/figures/distance_distributions.png`): per-model histograms show two modes — a small spike near 0 (mostly-overlapping ingredients) and a large mass at 0.6–1.0 (near-complete replacement). Models effectively choose between *keep* and *rewrite* rather than gradually substituting.

**Structural change is uniformly minimal.** `cosine_structural` mean is 0.07 across all 12 cells (range 0.049–0.112). Models update content but rarely restructure (meal layout / fitness skeleton stays the same). Worth one paragraph of paper discussion.

**Visualizations** ([src/plot_adaptivity.py](src/plot_adaptivity.py)):
- [results/figures/adaptivity_curves.png](results/figures/adaptivity_curves.png) (+ `.pdf`) — 2×2 panel of box plots with scatter overlay, one per signal, grouped by provider × category. Headline figure.
- [results/figures/distance_distributions.png](results/figures/distance_distributions.png) (+ `.pdf`) — Per-model step histograms of all four signals. Reveals the Jaccard bimodality.
- [results/adaptivity_summary.csv](results/adaptivity_summary.csv) — n / mean / std / median / IQR / min / max per (provider, category, signal). 48 rows. Goes in the paper as Table 2.

**Reproducibility:** all embeddings cached at `data/embeddings_cache.{keys.json, vectors.npy}`. The `embedding_model` field is recorded on every row of `results/similarity.csv` so future readers can verify which model produced the numbers (`sentence-transformers/all-MiniLM-L6-v2`, dim=384). Deterministic: same input → identical embeddings on every run.

**Repository scaffold:** complete. Source files for the four judges (Track A), arena evaluation (Track D), aggregation (Track E), and ML baseline (Track C, gated on Phase 6 labels) are stubbed in [src/](src/).

## Project structure

```
accessible-health-bench/
├── data/
│   ├── LLM_Prompts.csv               # source spreadsheet (120 rows)
│   ├── prompts.jsonl                 # converted, validated prompts
│   ├── responses/                    # raw model responses (480, gitignored)
│   ├── extractions/                  # structured JSON (480, gitignored)
│   ├── enriched/                     # extractions + grounding (480, gitignored)
│   ├── external/                     # BLS / USDA / Compendium reference CSVs
│   ├── llm_cache.sqlite              # cached API calls (gitignored)
│   └── wikidata_cache.sqlite         # cached SPARQL queries (gitignored)
├── src/
│   ├── config.py                     # model IDs, paths, generation params
│   ├── clients/unified_llm.py        # multi-provider LLM client w/ caching
│   ├── generate.py                   # Phase 2 — 480 responses
│   ├── extract.py                    # Phase 3 — structured extraction
│   ├── validate_extractions.py       # Phase 3 validator
│   ├── download_external_data.py     # Phase 4 prep — build reference CSVs
│   ├── grounding/
│   │   ├── wikidata.py               # SPARQL + LLM-fallback cuisine grounder
│   │   ├── bls.py                    # BLS staple-price grounder
│   │   ├── thrifty_plan.py           # USDA cost calibration grounder
│   │   └── compendium.py             # 2024 Compendium MET / WHO grounder
│   ├── ground_all.py                 # Phase 4 orchestrator (3-pass)
│   ├── coverage_report.py            # Phase 4 reporter — 5 paper CSVs
│   ├── similarity.py                 # Phase 5 Track B — 4 distance signals
│   ├── plot_adaptivity.py            # Phase 5 Track B — figures + diagnostic
│   ├── judges/                       # Phase 5 Track A — G-Eval / DAGMetric (TODO)
│   ├── arena_eval.py                 # Phase 5 Track D — pairwise matrix (TODO)
│   ├── ml_baseline.py                # Phase 5 Track C — logistic baseline (gated)
│   └── aggregate.py                  # Phase 5 Track E — combine into scores.csv (TODO)
├── prompts/                          # version-controlled judge prompt templates
│   └── extraction.txt                # Phase 3 extraction template
├── scripts/
│   ├── csv_to_jsonl.py               # CSV → JSONL converter
│   └── mini_pilot.py                 # Phase 4 6-prompt × 4-provider verifier
├── notebooks/                        # 01_eda, 02_results, 03_figures
├── dashboard/                        # React + Vite results viewer
├── results/
│   ├── coverage_report.csv           # Phase 4 per-response (480 rows)
│   ├── coverage_summary.csv          # Phase 4 aggregates (Table 1)
│   ├── thrifty_classification.csv    # Phase 4 — RQ1 cost buckets
│   ├── feasibility_assessment.csv    # Phase 4 — RQ3 WHO buckets
│   ├── cuisine_distribution.csv      # Phase 4 — RQ2 top cuisines
│   ├── similarity.csv                # Phase 5 Track B (240 rows)
│   ├── adaptivity_summary.csv        # Phase 5 Track B (Table 2)
│   ├── figures/
│   │   ├── adaptivity_curves.png     # Phase 5 Track B — headline figure
│   │   └── distance_distributions.png# Phase 5 Track B — bimodality figure
│   ├── constrained/                  # constrained-only Phase 4 CSVs
│   └── baseline/                     # baseline-only Phase 4 CSVs
└── paper/main.tex
```

## Roadmap

- **Phase 5 (current):**
  - ✅ Track B (Sentence-BERT similarity) complete — see results above.
  - ⬜ Track A: write `prompts/judge_*.txt` templates + implement `src/judges/*.py` + `src/run_judges.py` orchestrator. ~25 min, ~$3.50 to run on full 480.
  - ⬜ Track D: implement [src/arena_eval.py](src/arena_eval.py) for the 4×4 model-vs-model preference matrix on a 30-prompt subset. ~10 min, ~$0.50.
  - ⬜ Track E: implement [src/aggregate.py](src/aggregate.py) to join Tracks A/B/D + Phase 4 coverage into `results/scores.csv`.
  - ⬜ Track C (logistic regression baseline) — gated on Phase 6 manual-validation labels.
- **Phase 6 — Analysis:** Cohen's kappa inter-rater validation against a 30-sample manual set; logistic regression trained on those labels; figures; React dashboard build; paper draft.

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

Build external reference data (one-time, ~30 sec; persists for 30 days):
```
python -m src.download_external_data
```

Smoke-test each grounder individually before scaling:
```
python -m src.grounding.wikidata --test rice quinoa egusi salmon dal kichari kimchi
python -m src.grounding.bls       --test rice "ground beef" eggs salt
python -m src.grounding.thrifty_plan --household-type single
python -m src.grounding.compendium --test "push-ups" "running 6 mph" yoga "irish step dance" salt
```

Run the targeted 6-prompt mini-pilot (verifies all three RQ pipelines fire end-to-end):
```
python scripts/mini_pilot.py
```

Run the full Phase 4 grounding pass (~1 hour first time; restart-safe):
```
python -m src.ground_all
```

Run a Phase 4 grounding pilot (5 extractions per provider):
```
python -m src.ground_all --pilot 5
```

Generate the 5 paper-ready CSVs (split by variant):
```
python -m src.coverage_report
python -m src.coverage_report --variant constrained --out-dir results/constrained
python -m src.coverage_report --variant baseline    --out-dir results/baseline
```

Phase 5 Track B — compute Sentence-BERT distance signals:
```
python -m src.similarity --pilot 5     # 5 pair_keys × 4 providers — verify
python -m src.similarity               # full 240-pair run (~5 min cold, ~30s warm cache)
```

Phase 5 Track B — plot the Adaptivity Curve and distance distributions:
```
python -m src.plot_adaptivity
```
