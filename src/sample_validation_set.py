"""
src/sample_validation_set.py

Phase 6, Step 1 — Pick 30 responses for human scoring.

We need a small, balanced sample that lets us check whether our LLM judges
agree with humans. The sample is stratified so it covers:
  - all 4 providers (openai / anthropic / deepseek / groq)
  - all 3 categories (financial / cultural / lifestyle)
  - both variants (baseline / constrained), weighted toward constrained
    because that's where adherence judging actually fires

Outputs two CSV templates with empty score columns. Each rater fills in
their own copy independently. Then compute_kappa.py reads both back.

Run from repo root:
    python -m src.sample_validation_set
    python -m src.sample_validation_set --n 30 --seed 42
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS, JUDGED_DIR, RESULTS_DIR


# Fraction of the sample drawn from constrained prompts (rest from baselines).
CONSTRAINED_FRACTION = 0.7

# Where the templates are written.
TEMPLATE_DIR = Path(RESULTS_DIR) / "validation"


def load_judged_records(providers: list[str]) -> list[dict]:
    """Walk data/judged/ and return one summary row per response."""
    rows = []
    for provider in providers:
        provider_dir = Path(JUDGED_DIR) / provider
        if not provider_dir.exists():
            continue
        for f in sorted(provider_dir.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "_skipped" in (rec.get("judges") or {}):
                continue
            rows.append({
                "provider":      provider,
                "prompt_id":     rec.get("prompt_id"),
                "category":      rec.get("category"),
                "variant":       rec.get("variant"),
                "category_type": rec.get("category_type") or "",
                "prompt_text":   rec.get("prompt_text", ""),
                "response_text": rec.get("response_text", ""),
            })
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Stratify by (category, variant). Roughly n*CONSTRAINED_FRACTION come
    from constrained prompts, balanced across the 3 categories. The
    remaining slots come from baselines, also balanced. Each cell additionally
    cycles through providers so all 4 models appear roughly equally."""
    rng = random.Random(seed)

    # Bucket rows by (category, variant)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r["category"], r["variant"])].append(r)

    n_constrained = round(n * CONSTRAINED_FRACTION)
    n_baseline = n - n_constrained

    categories = ("financial", "cultural", "lifestyle")
    per_cat_constrained = n_constrained // len(categories)
    per_cat_baseline = n_baseline // len(categories)

    # Top up the first category if there's a remainder.
    extra_constrained = n_constrained - per_cat_constrained * len(categories)
    extra_baseline = n_baseline - per_cat_baseline * len(categories)

    chosen: list[dict] = []
    for i, cat in enumerate(categories):
        for variant, base_count, extra in (
            ("constrained", per_cat_constrained, extra_constrained if i == 0 else 0),
            ("baseline",    per_cat_baseline,    extra_baseline if i == 0 else 0),
        ):
            target = base_count + extra
            pool = list(buckets.get((cat, variant), []))
            rng.shuffle(pool)
            # Cycle providers so we don't pick all openai responses by accident.
            picked: list[dict] = []
            seen_provider_count: dict[str, int] = defaultdict(int)
            for row in pool:
                if len(picked) >= target:
                    break
                # Prefer rows from a provider we've used least so far.
                # A simple greedy filter is fine at this scale.
                if seen_provider_count[row["provider"]] < (target // 4 + 1):
                    picked.append(row)
                    seen_provider_count[row["provider"]] += 1
            # Fallback: if we couldn't fill via provider-balance, fill with the
            # remaining shuffled rows.
            for row in pool:
                if len(picked) >= target:
                    break
                if row not in picked:
                    picked.append(row)
            chosen.extend(picked)

    rng.shuffle(chosen)
    return chosen[:n]


def write_template(rows: list[dict], path: Path, rater_name: str):
    """Write a CSV template with empty score columns for one rater."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_id", "rater",
        "provider", "prompt_id", "category", "variant", "category_type",
        "prompt_text", "response_text",
        # Score columns the rater fills in (1-5 integers; leave blank if N/A)
        "human_affordability",
        "human_cultural",
        "human_feasibility",
        "human_health_accuracy",
        # DAG-style adherence verdicts the rater fills in (yes / partial / no /
        # not_applicable). The "branch" matches the LLM adherence judge.
        "human_adherence_b1_financial",
        "human_adherence_b2_cultural",
        "human_adherence_b3_lifestyle",
        # Free-text notes
        "human_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(rows):
            writer.writerow({
                "row_id":       f"v{i:02d}",
                "rater":        rater_name,
                "provider":     r["provider"],
                "prompt_id":    r["prompt_id"],
                "category":     r["category"],
                "variant":      r["variant"],
                "category_type": r["category_type"],
                "prompt_text":  r["prompt_text"],
                "response_text": r["response_text"],
                "human_affordability":          "",
                "human_cultural":               "",
                "human_feasibility":            "",
                "human_health_accuracy":        "",
                "human_adherence_b1_financial":  "",
                "human_adherence_b2_cultural":   "",
                "human_adherence_b3_lifestyle":  "",
                "human_notes":                  "",
            })
    print(f"  wrote {path} ({len(rows)} rows)")


def print_sample_summary(rows: list[dict]):
    by_cell: dict[tuple, int] = defaultdict(int)
    for r in rows:
        by_cell[(r["provider"], r["category"], r["variant"])] += 1
    print(f"\nSample composition (n={len(rows)}):")
    print(f"  {'provider':10} {'category':10} {'variant':12} {'n':>3}")
    for key, n in sorted(by_cell.items()):
        prov, cat, var = key
        print(f"  {prov:10} {cat:10} {var:12} {n:>3}")


def main():
    parser = argparse.ArgumentParser(description="Sample validation set for Phase 6")
    parser.add_argument("--n", type=int, default=30,
                        help="Total number of responses to sample (default 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    parser.add_argument("--rater-names", nargs="+",
                        default=["sanjana", "snigdha"],
                        help="Names for the two rater CSV files")
    args = parser.parse_args()

    if not Path(JUDGED_DIR).exists():
        print(f"ERROR: {JUDGED_DIR} does not exist. Run `python -m src.run_judges` "
              "first.")
        return

    print(f"Loading judged records...")
    rows = load_judged_records(list(MODELS.keys()))
    print(f"  {len(rows)} total judged responses available")

    if len(rows) < args.n:
        print(f"ERROR: only {len(rows)} responses available; need {args.n}.")
        return

    sample = stratified_sample(rows, n=args.n, seed=args.seed)
    print_sample_summary(sample)

    # Write one template per rater. Both files have IDENTICAL rows (same row_id)
    # so compute_kappa.py can join them on row_id.
    for name in args.rater_names:
        out_path = TEMPLATE_DIR / f"human_scores_{name}.csv"
        write_template(sample, out_path, rater_name=name)

    # Also save the sample manifest so we know which prompts were picked.
    manifest_path = TEMPLATE_DIR / "validation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({
            "n": args.n,
            "seed": args.seed,
            "rater_names": args.rater_names,
            "row_ids": [f"v{i:02d}" for i in range(len(sample))],
            "prompt_ids": [
                f"{r['provider']}/{r['prompt_id']}" for r in sample
            ],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  wrote {manifest_path}")

    print("\nNext steps:")
    print("  1. Each rater fills in their own CSV INDEPENDENTLY.")
    print("     Don't discuss scores until BOTH CSVs are complete.")
    print("  2. Use prompts/human_scoring_guide.md as the rubric reference.")
    print("  3. Run: python -m src.compute_kappa")


if __name__ == "__main__":
    main()
