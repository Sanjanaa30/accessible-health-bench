# AccessibleHealthBench

> A reproducible benchmark testing whether four major Large Language Models adapt their nutrition and fitness advice when users state real-world constraints — financial, cultural, and lifestyle.

**Authors:** Sanjana Shivanand · Sai Snigdha Nadella · Binghamton University

**Snapshot:** 2026-04-29 — covering Phases 1 through 7

**Total project cost:** under $5 in API spend across 480 LLM responses + judging + arena comparisons.

---

## What this project asks

Many people now use AI chatbots for diet and exercise advice. But these chatbots may quietly assume the user has lots of money, a flexible schedule, a fully equipped kitchen, and follows a Western diet. We test whether they actually adapt when users tell them otherwise.

We focus on three concrete questions:

- **RQ1 — Financial accessibility:** When a user says they have $30/week for groceries, do models actually swap salmon for lentils, mention SNAP/WIC, and stay within the budget?
- **RQ2 — Cultural bias:** When a user says they cook traditional South Indian food, do models engage substantively (sambar, dal, tadka), or default to Western food with a token cultural mention?
- **RQ3 — Lifestyle constraints:** When a user says they work 12-hour shifts and have 20 minutes to cook, do models actually deliver 20-minute meals, or produce a generic 7-day plan?

---

## What we built

A six-phase reproducible pipeline running on **120 prompts × 4 LLMs = 480 responses**, plus visualizations in Section 7 of the report.

```
Phase 1: Dataset (120 prompts, 60 base × 2 variants)
   ↓
Phase 2: Generation (480 responses, cached)
   ↓
Phase 3: Extraction (GPT-4o-mini → 10-block JSON schema)
   ↓
Phase 4: Grounding (Wikidata + BLS + USDA + Compendium)
   ↓
Phase 5: Evaluation (5 tracks: judges, Sentence-BERT, logistic regression, ArenaGEval, aggregation)
   ↓
Phase 6: Human validation (N=15, both authors)
   ↓
Section 7: Visualizations (8 paper-quality figures + scorecard)
```

---

## Phase-by-phase summary (in plain language)

### Phase 1 — Dataset Construction (complete)

We hand-authored **120 prompts** representing realistic users. Each prompt has a baseline (no constraint) and a constrained version (constraint added). The pairing is what lets us compare: did the model change its answer when the constraint was introduced?

| Category   | Baseline | Constrained | Total |
|------------|----------|-------------|-------|
| Financial  | 20       | 20          | 40    |
| Cultural   | 20       | 20          | 40    |
| Lifestyle  | 20       | 20          | 40    |
| **Total**  | **60**   | **60**      | **120** |

Example pair:
- `fin_base_01`: "Suggest a healthy breakfast."
- `fin_con_01`: "Suggest a healthy breakfast. I have $5."

Outputs: [data/prompts.jsonl](data/prompts.jsonl) · [data/LLM_Prompts.csv](data/LLM_Prompts.csv) · [scripts/csv_to_jsonl.py](scripts/csv_to_jsonl.py)

### Phase 2 — Generation (complete, 480/480)

We sent every one of the 120 prompts to four large language models. Each call goes through a **unified multi-provider client** with SQLite caching, so we never pay for the same call twice.

**Models used (April 2026):**

| Provider  | Model ID                          | Display name      | Saved |
|-----------|-----------------------------------|-------------------|-------|
| OpenAI    | `gpt-4o-mini`                     | GPT-4o-mini       | 120 / 120 |
| Anthropic | `claude-haiku-4-5-20251001`       | Claude Haiku 4.5  | 120 / 120 |
| DeepSeek  | `deepseek-v4-flash`               | DeepSeek V4 Flash | 120 / 120 |
| Groq      | `llama-3.3-70b-versatile`         | Llama 3.3 70B     | 120 / 120 |

Generation params: `temperature=0.7`, `max_tokens=1500` (uniform across providers).

**Provider lineup change:** Gemini was originally part of the lineup but was replaced with DeepSeek mid-run because the free-tier quota was too restrictive and `gemini-1.5-flash` was retired during the run. The 20 partial Gemini responses were discarded.

Outputs: [src/clients/unified_llm.py](src/clients/unified_llm.py) · [src/generate.py](src/generate.py) · [src/config.py](src/config.py) · `data/responses/` (committed — 480 raw responses)

### Phase 3 — Structured Extraction (complete, 480/480)

Each free-text response was converted by GPT-4o-mini (temperature=0, max_tokens=8000) into a **strict 10-block structured JSON**. This makes downstream scoring tractable — judges read fields like `cost_information.total_cost_usd` instead of free prose.

The 10 schema blocks: `summary`, `response_type`, `primary_goal`, `meal_components`, `all_ingredients`, `all_dishes_or_foods_named`, `fitness_components`, `routine_structure`, `cost_information`, `cultural_signals`, `feasibility_signals`, `household_and_demographic_context`, `medical_or_health_signals`, `constraint_adherence`, `caveats_and_disclaimers`.

We had to raise `max_tokens` twice (2500 → 4000 → 8000) because dense 7-day plans can produce 14k+ characters of JSON. Final pass: 478 cleanly-parsed files + 2 advisory-style outputs flagged as suspicious-but-valid.

Outputs: [src/extract.py](src/extract.py) · [src/validate_extractions.py](src/validate_extractions.py) · [prompts/extraction.txt](prompts/extraction.txt) · `data/extractions/` (committed — 480 structured JSONs)

### Phase 4 — External Grounding (complete, 480/480)

We enriched each extracted record against **four authoritative external sources** so judges and classifiers can reason against real numbers, not just LLM opinions.

| Source | Module | Purpose | RQ |
|---|---|---|---|
| Wikidata SPARQL + LLM fallback | [src/grounding/wikidata.py](src/grounding/wikidata.py) | Cuisine origin tags | RQ2 |
| BLS Average Retail Food Prices | [src/grounding/bls.py](src/grounding/bls.py) | Per-ingredient unit prices | RQ1 |
| USDA Cost of Food at Home | [src/grounding/thrifty_plan.py](src/grounding/thrifty_plan.py) | Household-level weekly cost benchmarks | RQ1 |
| 2024 Adult Compendium of Physical Activities | [src/grounding/compendium.py](src/grounding/compendium.py) | MET energy values + WHO 2020 weekly compliance | RQ3 |

**Phase 4 coverage outcomes (averaged across applicable responses):**

| Metric | Constrained (n=240) | Baseline (n=240) | Notes |
|---|---|---|---|
| Wikidata cuisine coverage | 52.4% | 60.1% | Constrained prompts add region-specific items |
| BLS price coverage | 6.6% | 4.8% | Thin — BLS list has only 25 staples; international items often missed |
| Compendium fitness coverage | 19.4% | 12.4% | Of fitness-bearing responses (89 / 86 respectively) |
| Western-centricity ratio | 3.8% | 4.2% | Constrained prompts produce slightly LESS Western content |
| Any cost mentioned | 13% (of 240) | 0% | RQ1-specific quantification rate is **32.5%** of financial-constrained — see RQ1 results below |

**Three-pass orchestrator** in [src/ground_all.py](src/ground_all.py):
1. Ground each extraction; Wikidata accumulates misses across the run.
2. One batched LLM-fallback call resolves all accumulated cuisine misses (one call instead of N).
3. Finalize and write all 480 enriched files.

Outputs: [src/ground_all.py](src/ground_all.py) · [src/coverage_report.py](src/coverage_report.py) · [src/download_external_data.py](src/download_external_data.py) · `data/enriched/summary.json` (committed; per-record JSONs gitignored — regenerable from responses + grounding scripts) · `data/external/` (committed reference data)

### Phase 5 — Evaluation (complete, all 5 tracks)

Phase 5 ran five complementary evaluation tracks on the 480 enriched responses. Total Phase 5 cost: **~$2.30** (judges $1.49 initial + $0.79 rerun after rubric fix; arena $0.30; Sentence-BERT $0; aggregation $0).

#### Track A — LLM-as-Judge Scoring

Four GPT-4o-mini judges scored each response 1-5 on **affordability, cultural appropriateness, lifestyle feasibility**, plus per-RQ yes/partial/no adherence verdicts. We discovered and fixed two rubric bugs during pilot runs (baseline-vs-constrained branch confusion + cross-firing between dimensions). Final result: absolute scores cluster very tightly across providers, especially on subjective dimensions.

| Provider | Affordability | Cultural | Feasibility | Adherence |
|---|---|---|---|---|
| DeepSeek V4 Flash | 4.97 | 5.00 | 4.32 | 4.92 |
| Anthropic Haiku 4.5 | 4.83 | 4.99 | 4.25 | 4.92 |
| GPT-4o-mini | 4.80 | 4.96 | 4.18 | 4.84 |
| Llama 3.3 70B (Groq) | 4.88 | 4.97 | 4.08 | 4.95 |

**Note the ceiling effect:** 94–98% of constrained responses scored 5/5 on cultural and adherence. Feasibility was the only dimension where absolute scoring produced a real distribution (33.8% / 51.2% / 14.2% at scores 5 / 4 / 3 on constrained-lifestyle).

#### Track B — Sentence-BERT Similarity

For every (baseline, constrained) prompt pair, we measured how much the response changed using four distance signals: full-text cosine, ingredient-list cosine, structural-digest cosine, and Jaccard set distance. 240 paired comparisons.

**Headline result — adaptation magnitude (cosine_full mean):**

| Provider | Cultural | Financial | Lifestyle |
|---|---|---|---|
| Anthropic Haiku 4.5 | **0.369** | 0.278 | 0.272 |
| Llama 3.3 70B (Groq) | 0.281 | 0.286 | 0.239 |
| DeepSeek V4 Flash | 0.282 | 0.205 | 0.193 |
| GPT-4o-mini | 0.270 | 0.206 | **0.154** |

Spread between the most-adaptive and least-adaptive cell: **2.4×**. Models do NOT all respond to constraints equally.

**Bimodality finding:** Jaccard distance is bimodal across every model — a small spike near 0 (mostly-overlapping ingredients) and a large mass at 0.6–1.0 (near-complete replacement). Models switch between "keep" and "rewrite" modes rather than gradually substituting.

#### Track C — Logistic Regression Baseline

A small interpretable classifier was trained on five non-LLM features (cosine_full, cosine_ingredients, jaccard_ingredients, Western-centricity ratio, response-length ratio) to predict whether human raters considered a response "adherent" — using leave-one-out cross-validation at N=15.

| Target dimension | N | Class balance | Accuracy | Top feature (β) |
|---|---|---|---|---|
| Affordability | 15 | 13 / 2 (imbalanced) | 73.3% (CI 48–89%) | cosine_full (−0.34) |
| **Cultural** | 15 | 9 / 6 (balanced) | 73.3% (CI 48–89%) | **western_centricity (−1.07)** |
| Feasibility | 15 | 15 / 0 (no negative class) | Cannot train | n/a (no class-0) |

**The strongest single quantitative finding in the project:** the **Wikidata-derived Western-centricity ratio is the dominant predictor of cultural non-adherence (β = −1.07, more than 2× any other feature)**. The negative direction means: more Western content → less cultural adherence. This validates that the Phase 4 grounding pipeline produces a real, downstream-useful, interpretable signal.

#### Track D — ArenaGEval Pairwise Comparison

We presented every pair of model responses to GPT-4o-mini side-by-side on a stratified 60-prompt subset (20 per category, all constrained). Position randomized deterministically per (prompt, pair, dimension) so reruns hit the LLM cache. **1,080 comparisons total at $0.30**.

**Per-provider win rates (aggregated across all 6 model pairs):**

| Provider | Affordability | Cultural | Feasibility |
|---|---|---|---|
| DeepSeek V4 Flash | **78.9%** | **83.3%** | **87.2%** |
| Anthropic Haiku 4.5 | 63.7% | 36.7% | 51.7% |
| GPT-4o-mini | 32.2% | 50.0% | 41.7% |
| Llama 3.3 70B (Groq) | 24.7% | 29.8% | 19.4% |

**Pairwise comparison reveals what absolute scoring hides.** Same data, different evaluation method: absolute scores differ by 0.04 points on cultural; pairwise differs by **53.5 percentage points** on cultural. The provider ranking — **DeepSeek > Anthropic > OpenAI > Groq** — is consistent across all three dimensions.

#### Track E — Aggregation

All Phase 5 signals plus Phase 4 grounding metrics joined into [results/scores.csv](results/scores.csv) (480 rows, one per response) plus three derived summary tables.

Outputs: [src/run_judges.py](src/run_judges.py) · [src/judges/](src/judges/) · [src/similarity.py](src/similarity.py) · [src/ml_baseline.py](src/ml_baseline.py) · [src/arena_eval.py](src/arena_eval.py) · [src/aggregate_scores.py](src/aggregate_scores.py)

### Phase 6 — Human Validation (complete, N=15)

Both authors independently scored a stratified **15-response sample** (10 constrained + 5 baseline, balanced across providers and categories) on the same four dimensions the LLM judges produce, using a written rubric ([prompts/human_scoring_guide.md](prompts/human_scoring_guide.md)) that mirrors the judge rubrics. CSVs were submitted before discussing scores to keep ratings independent.

**Within-1-point agreement** is our main validation metric (more robust to ceiling effects than Cohen's kappa):

| Dimension | Inter-human within ±1 | LLM judge vs human-average within ±1 |
|---|---|---|
| Affordability | 73% (11/15) | **80%** (12/15) |
| Cultural | 40% (6/15) | 60% (9/15) |
| Feasibility | 87% (13/15) | **93%** (14/15) |

**The LLM judges agree with human ratings on most responses** — feasibility highest (93% within ±1, because it ties to concrete things like time and equipment), cultural lowest (60%, because it's the most subjective dimension where even the two humans agreed only 40%).

Cohen's kappa was computed for completeness (`results/kappa_report.csv`) but values were low (0.08–0.23) due to the ceiling effect — when humans rate everything 4 or 5, kappa is mathematically driven toward zero. Within-1-point is the meaningful validation metric here.

Outputs: [src/sample_validation_set.py](src/sample_validation_set.py) · [src/compute_kappa.py](src/compute_kappa.py) · [results/validation/](results/validation/) (committed) · [results/kappa_report.csv](results/kappa_report.csv) (committed)

### Section 7 — Visualizations (complete, 8 figures)

Eight paper-quality figures generated from real CSVs by [src/generate_phase7_figures.py](src/generate_phase7_figures.py). All figures saved as both PNG (display) and PDF (paper-quality vector) in `results/figures/` (committed — 8 PNG + 8 PDF).

| Figure | What it shows |
|---|---|
| **1A** | 80% ingredient replacement vs 32.5% cost quantification — visual evidence of "models change food but rarely commit to a price" |
| **1B** | Affordability ranking (DeepSeek 78.9% > Anthropic 63.7% > OpenAI 32.2% > Groq 24.7%) with Wilson 95% CIs |
| **2A** | Logistic regression coefficients for cultural target — Western-centricity dominates at β = −1.07 (>2× any other feature) |
| **2B** | Adaptation magnitude vs pairwise quality — Anthropic adapts most but DeepSeek wins more pairwise |
| **3A** | Stacked score distribution showing cultural/adherence pile at 5/5 (94–98%) while feasibility shows real distribution |
| **3B** | Grouped bars across 3 RQs × 4 providers — lifestyle has the widest spread |
| **5** | Phase 6 within-1-point agreement (inter-human + LLM-vs-human, three dimensions) |
| **6** | **Overall scorecard** — 4 × 3 heatmap of arena win rates with row + column averages |

---

## Combined Findings by Research Question

### RQ1 — Financial Accessibility

**Yes, but quality differs sharply across providers, and bigger changes are not necessarily better changes.**

- Models replace ~80% of ingredients when a budget is stated (Phase 5 Track B Jaccard distance).
- Only **32.5% (n = 26 of 80)** of financial-constrained responses commit to a quantified total cost we can check against the budget; the remaining 67.5% recommend foods without saying how much the plan would cost.
- When models DO quote a cost, it spans the full USDA Thrifty / Low / Moderate / Liberal range — some are SNAP-realistic, others quietly land 50%+ above the budget.
- Pairwise quality ranking: **DeepSeek 78.9% > Anthropic 63.7% > OpenAI 32.2% > Groq 24.7%**.
- Logistic regression: cosine_full distance is the top feature (β = −0.34, negative). Larger response rewrites correlate with WORSE human-rated affordability.
- Phase 6 within-1-point: 80% (judge vs human average).

### RQ2 — Cultural Bias

**Yes — and this is the project's clearest finding.**

- Western-centricity averages 3.8% on constrained vs 4.2% on baselines — models DO produce some non-Western content when asked.
- **The Wikidata Western-centricity ratio is the dominant predictor of cultural non-adherence (β = −1.07, more than 2× any other feature).** This validates the Phase 4 grounding pipeline as producing a real, downstream-useful signal: more Western content → less cultural adherence.
- Pairwise: **DeepSeek 83.3% > OpenAI 50.0% > Anthropic 36.7% > Groq 29.8%**.
- Anthropic adapts MOST aggressively in Sentence-BERT distance (cosine 0.369) yet ranks 3rd in pairwise cultural quality — change magnitude ≠ change quality.
- Phase 6 within-1-point: 60% (judge vs human). Cultural is the most subjective dimension; even the two humans agreed only 40%.

### RQ3 — Lifestyle Constraints

**Mostly yes, with notable quality gaps.**

- All four models meet a basic feasibility floor — no human-rated response in our sample scored below 2.5.
- But "feasible" ≠ "well-tailored." Only 33.8% of constrained-lifestyle responses scored a perfect 5; 14.2% landed in borderline (3) territory.
- Compendium MET coverage was only 19.4% — models often promise fitness with vague phrasings ("light cardio") rather than concrete activities.
- Pairwise: **DeepSeek 87.2% > Anthropic 51.7% > OpenAI 41.7% > Groq 19.4%** — the WIDEST provider gap of any RQ.
- Phase 6 within-1-point: **93%** (judge vs human, the highest agreement of any dimension — feasibility ties to concrete anchors like time and equipment).

### Overall — Bias and Accessibility Scorecard

The Section 7 scorecard summarizes everything in a single 4 × 3 grid (Figure 6):

| | Affordability | Cultural | Feasibility | Overall |
|---|---|---|---|---|
| **DeepSeek** | 78.9% | 83.3% | 87.2% | **83.1%** (uniformly green) |
| **Anthropic** | 63.7% | 36.7% | 51.7% | 50.7% (mixed) |
| **OpenAI** | 32.2% | 50.0% | 41.7% | 41.3% (mixed) |
| **Groq** | 24.7% | 29.8% | 19.4% | **24.6%** (uniformly red) |

**DeepSeek consistently produces the most accessible advice across every research question; Groq consistently produces the least accessible.** Anthropic and OpenAI are mixed — good on one dimension and weak on another.

---

## Main Limitations (5 disclosed)

1. **Small human-validation sample (N = 15).** We planned 30; trimmed to 15 for timeline. Per-RQ DAG-branch logistic regression couldn't train; we pivoted to dimension-level classifiers.
2. **Both human raters are project authors.** Convenience sample, not blind. Future work should recruit external raters.
3. **ArenaGEval pairwise judge is a single LLM (GPT-4o-mini).** Stylistic-alignment bias risk — DeepSeek and GPT-4o-mini may share preferences. Cross-validate with another LLM judge.
4. **BLS price coverage is thin (~7%).** Only 25 staples; international items (egusi, kichari, halloumi) miss. Affordability judge leans on USDA Thrifty Plan calibration.
5. **Absolute LLM-as-judge scoring is severely ceiling-pressed.** 94–98% scored 5/5 on cultural and adherence even after rubric fixes. Mitigated via pairwise + within-1-point, but absolute scoring on these dimensions cannot differentiate models.

---

## Project structure

```
accessible-health-bench/
├── data/
│   ├── LLM_Prompts.csv               # source spreadsheet (committed)
│   ├── prompts.jsonl                 # 120 prompts (committed)
│   ├── responses/                    # raw LLM responses (committed, 480)
│   ├── extractions/                  # structured JSON (committed, 480)
│   ├── enriched/summary.json         # aggregate enrichment summary (committed)
│   ├── enriched/*                    # per-record enriched files (gitignored — regenerable)
│   ├── judged/                       # LLM judge outputs (committed, 480)
│   ├── external/                     # BLS / USDA / Compendium reference CSVs (committed)
│   ├── *.sqlite                      # API + SPARQL caches (gitignored)
│   └── embeddings_cache.*            # Sentence-BERT cache (gitignored)
├── src/
│   ├── config.py                     # model IDs, paths, generation params
│   ├── clients/unified_llm.py        # multi-provider LLM client w/ caching
│   ├── generate.py                   # Phase 2
│   ├── extract.py                    # Phase 3
│   ├── validate_extractions.py       # Phase 3 validator
│   ├── download_external_data.py     # Phase 4 prep
│   ├── grounding/
│   │   ├── wikidata.py               # SPARQL + LLM-fallback cuisine grounder
│   │   ├── bls.py                    # BLS staple-price grounder
│   │   ├── thrifty_plan.py           # USDA cost calibration
│   │   └── compendium.py             # 2024 Compendium MET / WHO grounder
│   ├── ground_all.py                 # Phase 4 orchestrator (3-pass)
│   ├── coverage_report.py            # Phase 4 reporter — 5 paper CSVs
│   ├── similarity.py                 # Phase 5 Track B — Sentence-BERT
│   ├── plot_adaptivity.py            # Track B figures (legacy)
│   ├── judges/                       # Phase 5 Track A
│   │   ├── base.py                   # shared G-Eval judge runner
│   │   ├── affordability.py
│   │   ├── cultural.py
│   │   ├── feasibility.py
│   │   └── adherence.py              # DAGMetric judge
│   ├── run_judges.py                 # Track A orchestrator (resumable)
│   ├── ml_baseline.py                # Track C — logistic regression
│   ├── arena_eval.py                 # Track D — ArenaGEval pairwise
│   ├── aggregate_scores.py           # Track E — master scores.csv
│   ├── sample_validation_set.py      # Phase 6 sampler
│   ├── compute_kappa.py              # Phase 6 — Cohen's kappa + within-1
│   └── generate_phase7_figures.py    # Section 7 — 8 figures
├── prompts/                          # judge templates + scoring guide
│   ├── extraction.txt
│   ├── judge_affordability.txt
│   ├── judge_cultural.txt
│   ├── judge_adherence_dag.txt
│   ├── judge_feasibility.txt
│   └── human_scoring_guide.md        # Phase 6 rubric
├── scripts/
│   ├── csv_to_jsonl.py
│   ├── mini_pilot.py
│   └── generate_progress_report.py   # build the PDF report
├── results/
│   ├── *.csv                         # COMMITTED (scores, arena, similarity, kappa, ml_baseline, etc.)
│   ├── ml_baseline_folds_*.csv       # gitignored (per-fold detail; aggregate summary is committed)
│   ├── figures/                      # COMMITTED (8 PNG + 8 PDF)
│   ├── validation/                   # COMMITTED (human ratings + manifest)
│   ├── baseline/                     # COMMITTED (per-branch CSVs)
│   └── constrained/                  # COMMITTED (per-branch CSVs)
└── paper/
    ├── main.tex                      # placeholder for future LaTeX paper
    └── AccessibleHealthBench_progress_report.pdf   # COMMITTED — final deliverable
```

---

## Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1     # PowerShell on Windows
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

---

## How to run the full pipeline (in order)

```powershell
# Phase 1 — regenerate prompts (only if you edit the CSV)
python scripts/csv_to_jsonl.py

# Phase 2 — generation (~25 min on first run, instant on rerun via cache)
python -m src.generate

# Phase 3 — extraction (~10 min, uses cache)
python -m src.extract
python -m src.validate_extractions

# Phase 4 — grounding (one-time external data + ~1 hr first run)
python -m src.download_external_data
python -m src.ground_all
python -m src.coverage_report

# Phase 5 — evaluation
python -m src.similarity                   # Track B (~5 min cold, ~30s warm)
python -m src.run_judges                   # Track A (~25 min, ~$1.50)
python -m src.arena_eval                   # Track D (~30 min, ~$0.30)
python -m src.ml_baseline --target cultural  # Track C
python -m src.aggregate_scores             # Track E

# Phase 6 — human validation (manual rating step in the middle)
python -m src.sample_validation_set --n 15  # generates rater CSVs
# (each author fills in their CSV independently)
python -m src.compute_kappa                # computes agreement metrics

# Section 7 — visualizations
python -m src.generate_phase7_figures      # produces 8 figures

# Build the progress report PDF
python scripts/generate_progress_report.py
```

**Cost summary:** under $5 total across Phases 5 + 6.

---

## Key outputs to look at

- **[paper/AccessibleHealthBench_progress_report.pdf](paper/AccessibleHealthBench_progress_report.pdf)** — the headline deliverable, ~30 pages with all phases, findings, figures, limitations, and conclusion.
- **[results/scores.csv](results/scores.csv)** (committed) — master scores table, 480 rows × ~30 columns.
- **[results/figures/](results/figures/)** (committed) — 8 PNG + 8 PDF figures produced by `src/generate_phase7_figures.py`.
- **[results/validation/](results/validation/)** (committed) — human-rated CSVs + reproducibility manifest.
- **[data/external/MANIFEST.json](data/external/MANIFEST.json)** — SHA256 hashes pinning the BLS / USDA / Compendium reference data snapshot used in this report.

---

## Academic integrity note

Every artifact in this repository is real:
- All 480 LLM responses came from real API calls billed to our keys.
- All extractions, grounding queries, similarity computations, judge scores, and arena comparisons used real model output.
- Both human raters in Phase 6 are the project authors; we scored independently and submitted our CSVs before discussing.
- No data is fabricated; all limitations are disclosed in Section 8 of the report.
- A reviewer can reproduce every numerical claim in the report from the committed code + reference data + cache.
