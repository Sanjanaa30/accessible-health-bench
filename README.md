# AccessibleHealthBench

> A reproducible benchmark testing whether four major Large Language Models adapt their nutrition and fitness advice when users state real-world constraints — financial, cultural, and lifestyle.

**Authors:** Sanjana Shivanand · Sai Snigdha Nadella · Binghamton University

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

**Lineup design — recency as an independent variable.** The four providers were chosen to span a deliberate release-date range, so the project can ask not just *"which model is most accessibility-aware?"* but also *"are newer models meaningfully better at adapting to financial / cultural / lifestyle constraints than older ones?"* DeepSeek V4 Flash is the newest release in the lineup; Llama 3.3 70B is the oldest. The other two sit in between.

**Models used (run executed April 2026):**

| Provider  | Model ID                          | Display name      | Released   | Saved |
|-----------|-----------------------------------|-------------------|------------|-------|
| DeepSeek  | `deepseek-v4-flash`               | DeepSeek V4 Flash | Apr 2026   | 120 / 120 |
| Anthropic | `claude-haiku-4-5-20251001`       | Claude Haiku 4.5  | Oct 2025   | 120 / 120 |
| Groq      | `llama-3.3-70b-versatile`         | Llama 3.3 70B     | Dec 2024   | 120 / 120 |
| OpenAI    | `gpt-4o-mini`                     | GPT-4o-mini       | Jul 2024   | 120 / 120 |

Generation params: `temperature=0.7`, `max_tokens=1500` (uniform across providers).

**One thing to keep in mind.** Newer models also tend to be bigger and come from different labs with different training approaches. So we can't say "being newer" is the *only* reason a model performs better — size and training choices play a role too. What we *can* say: across all three research questions, the newer the model, the better it adapted to user constraints. This is discussed again in the Limitations section.

Outputs: [src/clients/unified_llm.py](src/clients/unified_llm.py) · [src/generate.py](src/generate.py) · [src/config.py](src/config.py) · `data/responses/` (committed — 480 raw responses)

### Phase 3 — Structured Extraction (complete, 480/480)

Each model's free-form answer was reorganized into a **fixed JSON record with 15 sections** — things like "all ingredients listed," "all dishes named," "cost information," "cultural signals," and so on. This step is just sorting; no opinions, no scoring. We use GPT-4o-mini for it (run at deterministic settings so the output is repeatable). The point is to make the next phases easy: a judge can look up "did the response give a total cost?" without re-reading the whole reply. There's also a small `extraction_notes` field where the extractor can flag anything that looked unclear.

**Why we use one fixed model to do all the checking.** The same GPT-4o-mini also acts as the LLM judge in Phase 5. Sorting text into a form and applying a rubric are both mechanical jobs — a newer model wouldn't change the output much. What matters is keeping the *checker* the same across every provider, so when DeepSeek and Llama get different scores, we know it's the responses that differ, not the thing scoring them. GPT-4o-mini also happens to be the oldest model in the lineup, which keeps the recency comparison clean.

The 15 sections: `summary`, `response_type`, `primary_goal`, `meal_components`, `all_ingredients`, `all_dishes_or_foods_named`, `fitness_components`, `routine_structure`, `cost_information`, `cultural_signals`, `feasibility_signals`, `household_and_demographic_context`, `medical_or_health_signals`, `constraint_adherence`, `caveats_and_disclaimers`.

We had to raise the output-length cap twice during piloting (2500 → 4000 → 8000 tokens) because some 7-day meal plans produced very long JSON. Final pass: 478 of 480 came back as clean JSON; the other 2 were short advisory-style answers, kept and flagged.

Outputs: [src/extract.py](src/extract.py) · [src/validate_extractions.py](src/validate_extractions.py) · [prompts/extraction.txt](prompts/extraction.txt) · `data/extractions/` (committed — 480 structured JSONs)

### Phase 4 — External Grounding (complete, 480/480)

We enriched each extracted record against **four authoritative external sources** so judges and classifiers can reason against real numbers, not just LLM opinions.

| Source | Module | Purpose | RQ |
|---|---|---|---|
| Wikidata SPARQL + LLM fallback | [src/grounding/wikidata.py](src/grounding/wikidata.py) | Cuisine origin tags | RQ2 |
| BLS Average Retail Food Prices | [src/grounding/bls.py](src/grounding/bls.py) | Per-ingredient unit prices | RQ1 |
| USDA Cost of Food at Home | [src/grounding/thrifty_plan.py](src/grounding/thrifty_plan.py) | Household-level weekly cost benchmarks | RQ1 |
| 2024 Adult Compendium of Physical Activities | [src/grounding/compendium.py](src/grounding/compendium.py) | MET energy values + WHO 2020 weekly compliance | RQ3 |

**How well did each external source actually cover what the models said?**

| What we measured | Constrained prompts (n=240) | Baseline prompts (n=240) | What this means |
|---|---|---|---|
| Dishes that Wikidata could tag with a cuisine | 52.4% | 60.1% | Constrained answers include more region-specific dishes (sambar, jollof, kichari) — those are real foods, but Wikidata indexes Western dishes more thoroughly |
| Ingredients with a BLS price match | 6.6% | 4.8% | The BLS list has only ~25 staples; international ingredients usually miss. We compensate using USDA Thrifty Plan calibration |
| Fitness activities with a Compendium MET value | 19.4% | 12.4% | Counted only within fitness-bearing responses (89 constrained / 86 baseline). Models often write vague phrases like "light cardio" instead of named activities |
| Western-centric content (share of dishes tagged Western) | 3.8% | 4.2% | Constrained answers actually have slightly LESS Western content — models DO shift away from Western defaults when asked |
| Responses that gave any cost number | 13% (of 240) | 0% | The RQ1-specific number — **32.5% of financial-constrained answers** — is the meaningful one. See RQ1 results below |

**How grounding ran (in three passes).** Defined in [src/ground_all.py](src/ground_all.py):

1. **Pass 1** — **Try the free source first.**
We loop through all 480 responses, pull out every dish name, and ask Wikidata: "What cuisine is this?" Whatever Wikidata answers, we save. Whatever Wikidata doesn't know (e.g., "egusi soup," "kichari," "ragi mudde") goes into a leftover pile.
2. **Pass 2** — **Ask the LLM only for the leftovers.**
Now we take that whole leftover pile and hand it to GPT-4o-mini in one batched message: "Here are 200 dishes Wikidata couldn't tag. What cuisine is each one?" GPT-4o-mini fills them in.
3. **Pass 3** — **Stitch everything together and save.**
Now we have two piles: the Wikidata answers from Pass 1 and the LLM answers from Pass 2. We merge them back into each response's record (so a single response file shows tags for all of its dishes — some from Wikidata, some from the LLM). Then we write all 480 final enriched JSON files to disk.

**Why this design:** start with the free source, only pay for what the free source doesn't know, and pay in one bulk call instead of hundreds of small ones. That's the entire idea.

Outputs: [src/ground_all.py](src/ground_all.py) · [src/coverage_report.py](src/coverage_report.py) · [src/download_external_data.py](src/download_external_data.py) · `data/enriched/summary.json` (committed; per-record JSONs gitignored — regenerable from responses + grounding scripts) · `data/external/` (committed reference data)

### Phase 5 — Evaluation (complete, all 5 tracks)

Phase 5 evaluated all 480 responses in **five different ways**.

#### Track A — LLM-as-Judge Scoring

We asked GPT-4o-mini to play the role of **four separate judges**, scoring each response on a 1-to-5 scale across three questions: *Was it affordable? Was it culturally appropriate? Was it doable for the user's lifestyle?* Plus a fourth yes/partial/no check on whether the response actually engaged with the user's stated constraint.

While piloting, we caught two bugs in how the rubrics were written and re-ran the judges after fixing them:

- **Bug 1:** judges were penalizing baseline responses for not addressing budgets the user never mentioned.
- **Bug 2:** the affordability judge was firing on cultural and lifestyle prompts — for example, scoring a Yom Kippur prompt 1/5 because the response didn't talk about money, even though money was never asked about.

After fixes, here are the average scores:

| Provider | Affordability | Cultural | Feasibility | Adherence |
|---|---|---|---|---|
| DeepSeek V4 Flash | 4.97 | 5.00 | 4.32 | 4.92 |
| Anthropic Haiku 4.5 | 4.83 | 4.99 | 4.25 | 4.92 |
| GPT-4o-mini | 4.80 | 4.96 | 4.18 | 4.84 |
| Llama 3.3 70B (Groq) | 4.88 | 4.97 | 4.08 | 4.95 |

**The numbers all look very close — and that's actually a problem.** This is called a "ceiling effect": when 94–98% of responses get the maximum 5/5 (cultural and adherence here), the scoring scale stops being able to tell the models apart. The one place scores actually spread out was lifestyle feasibility — about a third scored 5, half scored 4, and 14% landed at a borderline 3. To get real separation between providers we use the head-to-head comparison in Track D instead (described below).

**Could the judge itself be playing favorites?** Three reality-checks say **no** (the full discussion is in Limitation #3):

- **No self-preference.** If GPT-4o-mini secretly liked its own answers, GPT-4o-mini would have ranked first. Instead it came **3rd of 4** — that empirically rules out self-preference.
- **If anything, the top result is understated.** GPT-4o-mini is older than DeepSeek, so it may not always notice the more nuanced things DeepSeek does well. If that's happening, then DeepSeek's lead is actually *larger* than our numbers show — never smaller. So the result is safe in either case.
- **Humans agreed with the judge.** In Phase 6, two human raters hand-scored 15 responses. They agreed with the judge **within 1 point on 60–93% of cases**, depending on dimension. That's a real cross-check, not a hand-wave.

#### Track B — How much did each model actually change its answer?

**The question this track answers:** when a user adds a constraint to their prompt (e.g., *"…I have $30 a week"* or *"…I cook traditional South Indian food"*), how much does the model's answer actually change compared to its baseline answer? A little? A lot? Or barely at all?

**How we measured it.** For every prompt pair (baseline + constrained version), we asked **Sentence-BERT** to compare the two responses. Sentence-BERT is a small neural model that converts each piece of text into a numerical fingerprint — texts with similar meaning get similar fingerprints. We then computed a **distance score from 0 to 1** for each pair:

- `0.0` ≈ the two answers are almost identical (the model ignored the constraint and basically gave the same response twice)
- `0.5` ≈ noticeably different but recognizable as the same kind of plan (some real adaptation happened)
- `1.0` ≈ the two answers have almost nothing in common (the model rewrote everything)

We averaged this score across all 60 prompts in each (provider × category) cell. 

**Average rewrite size when a constraint was added:**

| Provider              | Cultural   | Financial | Lifestyle |
|-----------------------|------------|-----------|-----------|
| Anthropic Haiku 4.5   | **0.369**  | 0.278     | 0.272     |
| Llama 3.3 70B (Groq)  | 0.281      | 0.286     | 0.239     |
| DeepSeek V4 Flash     | 0.282      | 0.205     | 0.193     |
| GPT-4o-mini           | 0.270      | 0.206     | **0.154** |

Anthropic's average score on cultural prompts is `0.369`: *Anthropic did rewrite its answer when a cultural constraint was added, but only partly — somewhere between "kept most of it" and "wrote a whole new response."*

**Takeaway 1 — Which model changed the most and least** Anthropic, when a cultural constraint was added, scored 0.369 — the highest in the whole table. GPT-4o-mini, when a lifestyle constraint was added, scored 0.154 — the lowest. Therefore, Anthropic adapts about 2.4 times as much as GPT-4o-mini does, even though they're both responding to the same kind of "user added a constraint" change. **Models do not respond to user constraints equally** — and the gap is large enough to matter.

**Takeaway 2 — models adapt in an "all-or-nothing" pattern, not gradually.** We expected: when a model is asked to adapt, it would swap a few ingredients while keeping the rest of the meal plan. **That's not what we found.** When we lined up the baseline and constrained ingredient lists side by side for all 60 prompts per model, the results clustered into two groups:

  - **A small group where the two lists were nearly identical** — the model basically ignored the constraint and gave the same meal plan twice.
  - **A much bigger group where the two lists had almost nothing in common** — the model threw out its first answer and wrote a brand-new plan from scratch.
  - **Almost nothing in between.** The gentle "swap a couple of items" behavior we expected hardly ever happens.

**What this tells us.** When LLMs decide whether to adapt to a user's constraint, they treat it as a yes/no switch — either *"keep my original answer"* or *"start over and rewrite everything."* They don't seem to *edit* their recommendations; they replace them.

#### Track C — Can simple measurements predict what humans think?

We built a tiny prediction model that takes **five simple, mechanical measurements** about a response (no LLM judging involved) and tries to guess whether a human would rate the response as "good." The five inputs: how much the response changed when a constraint was added, how much the ingredient list changed, how much the ingredient sets overlapped, what fraction of the dishes were Western, and how much longer or shorter the constrained answer was. With only 15 hand-rated responses to work from, we trained using *leave-one-out* — train on 14, predict the 15th, repeat 15 times.

| Target dimension | N  | Class balance         | Accuracy           | Strongest predictor                                |
|------------------|----|-----------------------|--------------------|----------------------------------------------------|
| Affordability    | 15 | 13 / 2 (imbalanced)   | 73.3% (CI 48–89%)  | how much the whole response changed (β = −0.34)    |
| **Cultural**     | 15 | 9 / 6 (balanced)      | 73.3% (CI 48–89%)  | **fraction of Western dishes (β = −1.07)**         |
| Feasibility      | 15 | 15 / 0 (no negatives) | Can't train        | All 15 got "good" — no negatives to learn from     |

**The most important finding:**

- **One signal beat all the others by more than 2×.** Out of the five measurements, the *fraction of Western dishes in a response* was the single best predictor of whether a human would call that response culturally inappropriate — more than twice as useful as anything else we tried.
- **The pattern is simple and intuitive.** *The more Western a response is, the less likely a human is to call it culturally fitting.*
- **It's the project's strongest hard-numbers result.** Most of the other findings come from LLM judging or pairwise comparisons; this one comes from a real prediction model trained against real human ratings.
- **It validates the Phase 4 grounding work.** The "how Western is this?" score is produced fully automatically (Wikidata + LLM cuisine tagging) — no LLM judging needed. The fact that it predicts human judgment proves the Phase 4 pipeline produces a number with real downstream meaning, not just a nice-to-have.

#### Track D — Head-to-head model comparison

In Track A, we asked the judge to give each response a score on its own. The problem: almost everything scored 5/5, so the scores couldn't tell the models apart. Track D fixes that by switching the question — instead of *"how good is this answer?"* we ask *"which of these two answers is better?"*

**What we did:**

- Picked **60 prompts** (20 financial + 20 cultural + 20 lifestyle, all with constraints).
- For each prompt, took all four model responses and matched every model against every other model — that's 6 pairs (DeepSeek vs Anthropic, DeepSeek vs OpenAI, …).
- For each pair, asked GPT-4o-mini *"which response is better?"* on three dimensions: affordability, cultural fit, feasibility.
- Total: 60 prompts × 6 pairs × 3 dimensions = **1,080 head-to-head matches, all for $0.30**.
- Small detail: the order in which the two responses appear is shuffled (but reproducibly so), so the judge can't bias toward whichever one is shown first.

**Win rates — % of head-to-head matches each provider won:**

| Provider             | Affordability | Cultural    | Feasibility |
|----------------------|---------------|-------------|-------------|
| DeepSeek V4 Flash    | **78.9%**     | **83.3%**   | **87.2%**   |
| Anthropic Haiku 4.5  | 63.7%         | 36.7%       | 51.7%       |
| GPT-4o-mini          | 32.2%         | 50.0%       | 41.7%       |
| Llama 3.3 70B (Groq) | 24.7%         | 29.8%       | 19.4%       |

**Why this is the key result.** In Track A, all four models scored within 0.04 points of each other on cultural — basically a tie. **In Track D, the cultural gap is 53.5 percentage points** (DeepSeek 83.3% vs Groq 29.8%). Same responses, same judge — but a different *question* (compare two, instead of rate one) exposes a huge real difference between the models. The ranking — **DeepSeek > Anthropic > OpenAI > Groq** — holds across all three dimensions.

#### Track E — Aggregation

All Phase 5 signals + Phase 4 grounding metrics are joined into [results/scores.csv](results/scores.csv) (480 rows, one per response), plus three summary tables: [scores_summary.csv](results/scores_summary.csv), [scores_by_provider.csv](results/scores_by_provider.csv), and [scores_adherence_branches.csv](results/scores_adherence_branches.csv).

Outputs: [src/aggregate_scores.py](src/aggregate_scores.py)

### Phase 6 — Human Validation (complete, N=15)

Both authors hand-rated **15 responses** on the same 1-to-5 scale the LLM judges use, working from the same written rubric ([prompts/human_scoring_guide.md](prompts/human_scoring_guide.md)). The 15 responses were carefully picked to cover all four providers, all three categories (financial / cultural / lifestyle), and both branches (10 constrained + 5 baseline) — so they're a fair mini-sample of the full 480. Each author submitted their CSV **before discussing scores**, so the ratings stayed independent.

**The metric we use: "within ±1 point" agreement.** Two raters agree "within ±1" if their scores differ by 1 or less (4 vs 5 = agree; 3 vs 5 = disagree). It's a forgiving but standard agreement metric — it tolerates the natural difference between *"this looks like a 4"* and *"this looks like a 5"* without calling that disagreement.

| Dimension     | Author A vs Author B agree within ±1 | LLM judge vs human average agree within ±1 |
|---------------|--------------------------------------|---------------------------------------------|
| Affordability | 73% (11/15)                          | **80%** (12/15)                             |
| Cultural      | 40% (6/15)                           | 60% (9/15)                                  |
| Feasibility   | 87% (13/15)                          | **93%** (14/15)                             |

**What this tells us.** The LLM judge agrees with humans on most responses. Agreement is highest on **feasibility (93%)** because feasibility ties to concrete, checkable things like prep time and equipment. It's lowest on **cultural (60%)** because cultural appropriateness is genuinely the most subjective dimension — even the two human raters only agreed with each other 40% of the time on cultural.

**A note on Cohen's kappa.** Kappa is the standard *"are these two raters agreeing more than chance?"* statistic. We computed it for completeness ([results/kappa_report.csv](results/kappa_report.csv)), but the values came out low (0.08–0.23). That's a known quirk: when most responses get rated 4 or 5 (the ceiling effect from Track A), kappa is mathematically pushed toward zero even when the raters are visibly agreeing. So within-1-point is the meaningful number here.

Outputs: [src/sample_validation_set.py](src/sample_validation_set.py) · [src/compute_kappa.py](src/compute_kappa.py) · [results/validation/](results/validation/) (committed) · [results/kappa_report.csv](results/kappa_report.csv) (committed)

### Section 7 — Visualizations (complete, 8 figures)

Eight figures, each generated directly from the project's CSV results by [src/generate_phase7_figures.py](src/generate_phase7_figures.py). Two additional exploratory plots (`adaptivity_curves` and `distance_distributions`) are produced for diagnostic use.

The eight figures are grouped by research question, with two summary figures at the end:

| Figure | Tied to | What it shows |
|--------|---------|----------------|
| **1A** | RQ1 — Financial | When a user gives a budget, models change about **80% of the ingredients** in their answer — but only **32.5% of the time** do they actually say a total price. *They change the food, but rarely commit to a number.* |
| **1B** | RQ1 — Financial | A bar chart ranking the four models on how often they win head-to-head matches on affordability: DeepSeek 78.9% > Anthropic 63.7% > OpenAI 32.2% > Groq 24.7%. The small line on each bar shows the uncertainty range. |
| **2A** | RQ2 — Cultural | Five possible predictors of *"is this response culturally appropriate?"* drawn as bars. **"Fraction of Western dishes" is more than 2× the height of every other bar** — by far the strongest predictor. |
| **2B** | RQ2 — Cultural | A dot plot showing **how much** a model rewrites alongside **how good** the rewrite is. Anthropic rewrites the most, but DeepSeek wins more head-to-head matches — *changing more isn't the same as changing better.* |
| **3A** | RQ3 — Lifestyle | A breakdown of how the 1-to-5 scores fall across all 480 responses. Cultural and adherence are nearly always 5 (94–98%). Only feasibility shows a real spread — some 3s, some 4s, some 5s. |
| **3B** | RQ3 — Lifestyle | Bars grouped by research question and by provider. **Lifestyle is where the gap between best and worst provider is biggest.** |
| **5** | Phase 6 validation | Bars showing how often two raters agreed within 1 point — author A vs author B next to LLM judge vs human average — across affordability, cultural, and feasibility. |
| **6** | **Overall scorecard** | A coloured grid: 4 providers × 3 dimensions, green = high win rate, red = low. Row and column averages along the edges. **One picture that summarises the entire project.** |

---

## Combined Findings by Research Question

### RQ1 — Financial Accessibility

**Yes — models do change their answers when users mention a budget, but the *quality* of that change varies sharply across providers, and bigger rewrites aren't always better.**

- **Big food changes when a budget is given.** Models swap roughly **80% of the ingredients** in their answer compared to the no-budget version.
- **But they rarely commit to a price.** Only **32.5% (26 out of 80)** of budget-constrained responses actually quote a total cost the user can check against their budget. The remaining two-thirds list foods without saying what the plan would cost.
- **When a price IS quoted, it's all over the map.** Some land in the SNAP-realistic range; others quietly come in **50%+ over the user's stated budget** without flagging it.
- **Head-to-head ranking on affordability:** DeepSeek **78.9%** > Anthropic **63.7%** > OpenAI **32.2%** > Groq **24.7%**.
- **A counter-intuitive finding.** The *more* a model rewrites its answer, the *lower* humans rated it on affordability. Big rewrites don't mean better budgeting — if anything, the opposite.
- **Cross-checked with humans.** The LLM judge and human raters agreed within 1 point on **80%** of the affordability cases in Phase 6.

### RQ2 — Cultural Bias

**Yes — and this is the project's clearest finding.**

- **Models DO add some non-Western content when asked.** On cultural prompts, the share of dishes flagged as Western drops to **3.8%**, vs **4.2%** on the no-constraint baseline. A small but real shift in the right direction.
- **One signal towered over all the others.** The fraction of Western dishes in a response is **more than 2× as powerful** at predicting cultural non-adherence as any other measurement we tried. Translation: *the more Western a response is, the less likely a human is to call it culturally fitting.* This single number — produced automatically by the Phase 4 Wikidata grounding, with **no LLM judging at all** — actually predicts what humans care about.
- **Head-to-head ranking on cultural appropriateness:** DeepSeek **83.3%** > OpenAI **50.0%** > Anthropic **36.7%** > Groq **29.8%**.
- **A surprising twist.** Anthropic rewrites its answer the *most* aggressively when a cultural constraint is added (the highest rewrite-size score in the table at **0.369**), but it only ranks **3rd of 4** on cultural quality. **Changing more is not the same as changing better.**
- **Cross-checked with humans.** The LLM judge and human raters agreed within 1 point on **60%** of cultural cases — the lowest agreement of the three dimensions. But cultural is the most subjective overall: even the two human raters only agreed with each other 40% of the time.

### RQ3 — Lifestyle Constraints

**Mostly yes — with the biggest gaps between providers of any research question.**

- **A basic floor is met.** No human-rated response in our 15-sample scored below 2.5 on feasibility. Every provider clears at least the "workable advice" bar.
- **But "feasible" ≠ "well-tailored."** Only **33.8%** of lifestyle-constrained responses got a perfect 5; **14.2%** landed at a borderline 3 — the kind of advice that's *technically* doable but not really fitted to the user's actual situation.
- **Vague fitness language is everywhere.** Out of all responses that mentioned fitness, only **19.4%** named activities concrete enough for us to look up real energy/effort values. Models often hedge with phrases like *"light cardio"* or *"some movement"* instead of saying what the user should actually do.
- **Head-to-head ranking on lifestyle feasibility:** DeepSeek **87.2%** > Anthropic **51.7%** > OpenAI **41.7%** > Groq **19.4%**. **The widest provider gap of any research question** — almost 70 percentage points between best and worst.
- **Cross-checked with humans.** The LLM judge and human raters agreed within 1 point on **93%** of feasibility cases — the highest agreement of any dimension. That makes sense: feasibility ties to concrete, checkable things (prep time, equipment, schedule), which are easier to judge consistently than something subjective like cultural fit.

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

## Main Limitations
1. **Mixed-vintage provider lineup means "recency" cannot be cleanly isolated.** The four providers span April 2026 (DeepSeek V4 Flash) to July 2024 (GPT-4o-mini) — about 22 months. Recency was *intentional*, but it co-varies with model size, lab, and training philosophy. The *oldest* model (GPT-4o-mini, Jul 2024) outperforms the *second-newest* (Llama 3.3 70B, Dec 2024) in every Track D dimension, showing size and training matter on top of recency. Future work: re-run with same-vintage flagship models from each lab to disentangle these factors.

2. **All LLM judging in this project comes from a single judge model (GPT-4o-mini).** Best-practice benchmark papers run two judges and report agreement, or use a panel; we used one. Three things make this defensible rather than fatal: **(a)** if GPT-4o-mini favored its own outputs, GPT-4o-mini would have ranked first — instead it ranked 3rd of 4 on every dimension, which empirically refutes self-preference; **(b)** the concern that an older judge under-recognizes nuance in newer models would push DeepSeek's win *down*, not up — so the headline ranking is conservative under that concern, not inflated; **(c)** Phase 6 human raters cross-checked the judge on a stratified sample and agreed within ±1 point on **60% (cultural) / 80% (affordability) / 93% (feasibility)** of cases. The clean fix — adding a second LLM judge and reporting agreement — is the top item in future work.

3. **Hand-authored, English-only prompt set with limited cuisine coverage.** All 120 prompts are written by the two project authors in English, and the cultural slice covers a handful of cuisines (South Indian, West African Nigerian, Mediterranean, Levantine, East Asian, and a few others) rather than the full global range. Real users prompt LLMs in many languages and from many cultural backgrounds we don't represent. Future work: translate the prompt set, add native speakers from underrepresented cuisines as co-authors, and broaden cuisine coverage to at least 15-20 cultural traditions.
   
4. **Small human-validation sample (N = 15).** We originally planned 30; trimmed to 15 for timeline. Future work: scale to N ≥ 60 with raters from outside the project team.

5. **BLS price coverage is thin (~7%).** Only 25 staples are in the BLS list; international items (egusi, kichari, halloumi, kannadiga foods) routinely miss. Phase 4 mitigates this by falling back on USDA Thrifty Plan calibration for household-level cost benchmarks. Future work: integrate a broader retail-price dataset that includes international ingredients.
---

## Project structure

Only the items below are committed. Everything else (caches, virtualenvs, per-record enriched JSONs, figure PDFs, secrets, build artifacts) is in `.gitignore`.

```
accessible-health-bench/
├── data/
│   ├── LLM_Prompts.csv               # source spreadsheet (120 prompts)
│   ├── prompts.jsonl                 # 120 prompts in JSONL form
│   ├── responses/                    # 480 raw LLM responses
│   ├── extractions/                  # 480 structured JSONs (Phase 3)
│   ├── enriched/summary.json         # Phase 4 aggregate enrichment summary
│   ├── judged/                       # 480 LLM judge outputs (Phase 5 Track A)
│   └── external/                     # BLS / USDA / Compendium reference CSVs + MANIFEST.json
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
│   ├── coverage_report.py            # Phase 4 reporter
│   ├── similarity.py                 # Phase 5 Track B — Sentence-BERT
│   ├── judges/                       # Phase 5 Track A judges
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
│   └── csv_to_jsonl.py
├── results/
│   ├── *.csv                         # all aggregate result CSVs (scores, arena, similarity, kappa, ml_baseline summary, etc.)
│   ├── figures/                      # 8 PNG figures (PDFs regenerable from src/generate_phase7_figures.py)
│   ├── validation/                   # human ratings + reproducibility manifest
│   ├── baseline/                     # per-branch CSVs (no-constraint slice)
│   └── constrained/                  # per-branch CSVs (constrained slice)
```

**Not committed (gitignored, regenerable):** `data/*.sqlite` (LLM + Wikidata caches), `data/embeddings_cache.*` (Sentence-BERT vectors), `data/enriched/*` per-record JSONs, `results/ml_baseline_folds_*.csv` (per-fold detail), `results/figures/*.pdf`, `.env`, `.venv/`, `__pycache__/`.

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
   Add values for `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `GROQ_API_KEY`.

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

```

---

## Key outputs to look at

- **[results/scores.csv](results/scores.csv)** — master scores table, 480 rows × ~30 columns. Every claim in this README traces back to a row of this file.
- **[results/figures/](results/figures/)** — 10 PNG figures produced by [src/generate_phase7_figures.py](src/generate_phase7_figures.py); 8 are featured in the report, 2 are exploratory diagnostics.
- **[results/validation/](results/validation/)** — human-rated CSVs + reproducibility manifest from Phase 6.
- **[data/external/MANIFEST.json](data/external/MANIFEST.json)** — SHA256 hashes pinning the BLS / USDA / Compendium reference data snapshot.

---

## Academic integrity note

Every artifact in this repository is real:
- All 480 LLM responses came from real API calls billed to our keys.
- All extractions, grounding queries, similarity computations, judge scores, and arena comparisons used real model output.
- Both human raters in Phase 6 are the project authors; we scored independently and submitted our CSVs before discussing.
- No data is fabricated; all limitations are disclosed in Section 8 of the report.
