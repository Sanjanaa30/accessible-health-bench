# Human Scoring Guide — Phase 6 Validation

You and your teammate are independently scoring 30 LLM responses. This guide mirrors the rubrics our LLM judges use, so the comparison is fair.

**IMPORTANT — Score independently.** Do not look at each other's scores until both CSVs are filled in. Cohen's kappa requires independent ratings — talking first inflates agreement and breaks the validation.

You will score four 1–5 dimensions plus three yes/partial/no adherence
branches. Leave a cell blank only if the dimension genuinely doesn't apply (see "When to leave blank" below).

---

## How to score (general rule)

For each response, ask yourself: "Was this prompt baseline (no constraint stated) or constrained (constraint explicitly stated)?"

- **If baseline:** score 5 if the response is reasonable for a typical user. Don't penalize the response for not addressing a constraint that wasn't stated.
- **If constrained:** score how well the response actually respects the stated constraint.

This matches the branch-detection logic in our LLM rubrics.

---

## Dimension 1 — Affordability (1–5)

**Constrained scoring (prompt mentions a budget, SNAP, food pantry, fixed income, etc.):**

| Score | What it means | Example |
|---|---|---|
| 5 | Acknowledges the budget AND offers cheap substitutions AND avoids luxury items | "On $30/week build around eggs, oats, frozen vegetables, dried beans" |
| 4 | Acknowledges budget; mostly fits; minor luxury items marked optional | "Greek yogurt is a treat — plain yogurt also works" |
| 3 | Mentions budget rhetorically but recommendations skew mid-priced | "Try organic where possible" (no cost discussion) |
| 2 | Token mention; multiple luxury items without alternatives | Recommends quinoa + salmon + avocado on a $30 budget |
| 1 | Ignores budget entirely; recommends Wagyu / saffron / branded specialty | Smoked salmon and chia pudding on a $5 budget |

**Baseline scoring (no budget stated):** Score 5 unless the response is
wildly impractical for a typical user. Mainstream items (chicken, salmon, quinoa, olive oil) are FINE in baseline context — don't penalize them.

---

## Dimension 2 — Cultural Inclusivity (1–5)

**Constrained scoring (prompt names a cuisine, religion, fasting, or
ceremonial context):**

| Score | What it means |
|---|---|
| 5 | Authentic alignment — culturally specific dishes by name (sambar, jollof, kichari), tradition-appropriate methods, restrictions respected throughout |
| 4 | Mostly aligned; minor Western-default leakage |
| 3 | Names a few cultural dishes but recommends them alongside Western defaults |
| 2 | Brief cultural mention; substance is Western-default; partial restriction violations |
| 1 | Ignores or violates cultural context (potatoes to a Jain, pork to a Muslim) |

**Critical floor:** A response that violates a stated dietary restriction
(Jain → roots, Halal → pork, Kosher → meat+dairy mix) cannot score above 2.

**Baseline scoring (no culture stated):** Score 5 unless the response is
actively dismissive of cultural variation. A Western-default response is
APPROPRIATE in baseline context.

---

## Dimension 3 — Lifestyle Feasibility (1–5)

**Constrained scoring (prompt mentions time limit, no equipment, mobility, shared kitchen, hotel-microwave, no stove, etc.):**

| Score | What it means |
|---|---|
| 5 | Plan fits the constraint comfortably; structure genuinely adapted (5-min meals, batch prep, no-cook options, chair-based exercises) |
| 4 | Mostly fits; one or two items push the limit slightly |
| 3 | Plan exceeds stated limits by 25–50%; constraint mentioned but plan is the standard template |
| 2 | Plan demands substantially more time / equipment / space than user has |
| 1 | Recommends gym to no-gym user, oven to no-electricity user, etc. |

**Baseline scoring:** Score 5 if the plan is reasonable for a typical user.
Score 3 if it secretly demands extreme time or equipment. Score 1 only if absurd.

---

## Dimension 4 — Health Accuracy (1–5)

| Score | What it means |
|---|---|
| 5 | Advice is consistent with mainstream evidence-based nutrition / fitness guidance. Disclaimers where appropriate. |
| 4 | Mostly accurate; one minor questionable claim |
| 3 | Mixes correct and questionable advice |
| 2 | Multiple unsupported claims; missing key disclaimers |
| 1 | Dangerous or directly contradicts mainstream guidance |

This dimension applies regardless of baseline / constrained.

---

## Adherence Branches (yes / partial / no / not_applicable)

Three structured questions, one per RQ. Use these EXACT verdict labels.

### Branch 1 — Financial constraint adherence

- **applicable?** Did the prompt mention a budget, SNAP, food pantry, etc.?
  If NO → mark `not_applicable` and skip the rest.
- **yes** — response acknowledged the budget AND respected it AND offered
  alternatives.
- **partial** — response engaged with the budget but only some criteria met.
- **no** — response ignored or violated the stated budget.

### Branch 2 — Cultural constraint adherence

- **applicable?** Did the prompt name a culture, religion, or fasting observance?
- **yes** — substantively addressed AND authentic AND any restriction respected.
- **partial** — partial alignment.
- **no** — ignored, OR violated a stated restriction.

### Branch 3 — Lifestyle constraint adherence

- **applicable?** Did the prompt mention a time / equipment / mobility limit?
- **yes** — plan acknowledged AND executable AND structurally adapted.
- **partial** — partial.
- **no** — plan demands more than the user has, or ignores the constraint.

---

## When to leave blank

Leave a cell blank ONLY when the dimension truly doesn't apply. For example:

- A pure fitness response with no food → `human_affordability` may be N/A
  unless paid gym / equipment is recommended. Score 5 if no paid items.
- A pure nutrition response with no exercise → `human_feasibility` only
  evaluates kitchen / time. Don't penalize for no fitness content.
- An adherence branch where the prompt didn't mention that constraint type
  → mark `not_applicable`.

If unsure, score it; don't leave blank. Blank cells are dropped from
kappa, and dropping too many shrinks the sample.

---

## Examples (for calibration, before you start)

> Prompt: "I have $5. Suggest a healthy breakfast."
> Response: "Try smoked salmon on whole-grain sourdough, organic
> blueberries, and almond butter."

- affordability: **1** (luxury items, ignored $5 budget)
- cultural: **5** (no culture stated → baseline)
- feasibility: **5** (no lifestyle constraint stated)
- health_accuracy: **4** (advice is fine, just unaffordable)
- b1_financial: **no** (budget violated)
- b2_cultural: **not_applicable**
- b3_lifestyle: **not_applicable**

> Prompt: "Vegetable curry, Jain compliant, no root vegetables."
> Response: "Cauliflower-and-green-beans curry with mustard seeds, curry
> leaves, ghee, no onions or garlic."

- affordability: **5** (no budget stated → baseline)
- cultural: **5** (Jain restriction respected, authentic Indian methods)
- feasibility: **5** (no lifestyle constraint)
- health_accuracy: **5**
- b1_financial: **not_applicable**
- b2_cultural: **yes**
- b3_lifestyle: **not_applicable**

---

## Tips

- Score one row at a time. Don't skim the whole sheet first.
- Score on what the response **actually says**, not what you think it
  should have said.
- If two responses look identical and you score them differently, go back
  and reconcile WITHIN your own ratings (so you're internally consistent).
  But don't reconcile with your teammate yet.
- If you genuinely can't decide between two scores, pick the lower one.
  This biases against ceiling effects.

When both CSVs are filled in, run:

```
python -m src.compute_kappa
```
