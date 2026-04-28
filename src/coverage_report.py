"""
src/coverage_report.py

Phase 4 reporting — turn data/enriched/ into paper-ready CSVs.

Reads every enriched record produced by src/ground_all.py and emits:

  results/coverage_report.csv               — one row per response (raw)
  results/coverage_summary.csv              — provider x category aggregates
  results/thrifty_classification.csv        — distribution of cost buckets
  results/feasibility_assessment.csv        — distribution of WHO buckets
  results/cuisine_distribution.csv          — top cuisines per provider x category

Console output mirrors `data/enriched/_summary.json` so a single command
gives you the headline numbers.

Run from repo root:
    python -m src.coverage_report
    python -m src.coverage_report --provider openai
    python -m src.coverage_report --variant constrained
    python -m src.coverage_report --out-dir results/v2
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ENRICHED_DIR, MODELS, RESULTS_DIR


# =============================================================
# Dataclasses
# =============================================================
@dataclass
class CategoryAggregate:
    n: int = 0
    wd_coverage: list[float] = field(default_factory=list)
    bls_coverage: list[float] = field(default_factory=list)
    comp_coverage: list[float] = field(default_factory=list)
    western_centricity: list[float] = field(default_factory=list)
    thrifty_classes: Counter = field(default_factory=Counter)
    feasibility_classes: Counter = field(default_factory=Counter)
    cuisine_counts: Counter = field(default_factory=Counter)


# =============================================================
# Row extraction
# =============================================================
def extract_row(record: dict, provider: str) -> Optional[dict]:
    """Flatten one enriched record into a single CSV row dict.
    Returns None for skipped (extraction-error) records."""
    g = record.get("grounding", {}) or {}
    if "_skipped" in g:
        return None

    wd = g.get("wikidata", {}) or {}
    bls = g.get("bls_prices", {}) or {}
    thrifty = g.get("thrifty_plan", {}) or {}
    comp = g.get("compendium", {}) or {}

    comp_total = comp.get("total_count", 0)
    has_fitness = comp_total > 0

    return {
        "provider": provider,
        "prompt_id": record.get("prompt_id"),
        "category": record.get("category"),
        "variant": record.get("variant"),
        "category_type": record.get("category_type") or "",

        # Wikidata
        "wd_matched": wd.get("matched_count", 0),
        "wd_total": wd.get("total_count", 0),
        "wd_coverage": wd.get("coverage_ratio", 0.0),
        "western_centricity": wd.get("western_centricity_ratio"),  # may be None
        "western_tag_count": wd.get("western_tag_count", 0),
        "total_tag_count": wd.get("total_tag_count", 0),

        # BLS prices
        "bls_matched": bls.get("matched_count", 0),
        "bls_total_unique": bls.get("total_unique_ingredients", 0),
        "bls_coverage": bls.get("coverage_ratio", 0.0),

        # Thrifty plan
        "estimated_weekly_cost_usd": thrifty.get("response_estimated_cost_weekly_usd"),
        "cost_period_status": thrifty.get("response_cost_period_status", "n/a"),
        "thrifty_classification": thrifty.get("classification", "no_baseline"),
        "thrifty_ratio": thrifty.get("response_cost_vs_thrifty_ratio"),
        "household_size": thrifty.get("household_size"),
        "household_type": thrifty.get("household_type"),
        "thrifty_tier_usd": (thrifty.get("weekly_baseline_usd") or {}).get("thrifty"),

        # Compendium fitness
        "has_fitness_content": has_fitness,
        "comp_matched": comp.get("matched_count", 0),
        "comp_total": comp_total,
        "comp_coverage": comp.get("coverage_ratio", 0.0),
        "total_logged_minutes": comp.get("total_logged_minutes", 0),
        "moderate_equivalent_minutes": comp.get("moderate_equivalent_minutes", 0),
        "time_horizon": comp.get("time_horizon"),
        "feasibility_assessment": comp.get(
            "feasibility_assessment", "no_fitness_content"
        ),
    }


# =============================================================
# Aggregation
# =============================================================
def collect_rows(
    enriched_root: Path,
    providers: list[str],
    variant_filter: Optional[str],
    skipped_counter: Counter,
) -> list[dict]:
    rows: list[dict] = []
    for provider in providers:
        provider_dir = enriched_root / provider
        if not provider_dir.exists():
            print(f"  [skip] no enriched dir for {provider}")
            continue

        for f in sorted(provider_dir.glob("*.json")):
            # Defensive — skip MANIFEST-style sidecars at the provider level
            if f.name.startswith("_"):
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                skipped_counter["unreadable"] += 1
                continue

            row = extract_row(rec, provider)
            if row is None:
                skipped_counter["extraction_error"] += 1
                continue

            if variant_filter and row["variant"] != variant_filter:
                continue

            rows.append(row)
    return rows


def build_aggregates(rows: list[dict]) -> dict[tuple, CategoryAggregate]:
    """Group by (provider, category) and accumulate stats."""
    agg: dict[tuple, CategoryAggregate] = defaultdict(CategoryAggregate)
    for row in rows:
        key = (row["provider"], row["category"])
        a = agg[key]
        a.n += 1
        a.wd_coverage.append(row["wd_coverage"])
        a.bls_coverage.append(row["bls_coverage"])

        # Only count Compendium coverage for fitness-bearing responses
        if row["has_fitness_content"]:
            a.comp_coverage.append(row["comp_coverage"])
            fb = row["feasibility_assessment"]
            if fb:
                a.feasibility_classes[fb] += 1

        # Preserve the distinction between None (no Wikidata content) and 0.0
        if row["western_centricity"] is not None:
            a.western_centricity.append(row["western_centricity"])

        cls = row["thrifty_classification"]
        if cls:
            a.thrifty_classes[cls] += 1

    return agg


# =============================================================
# Loading per-record cuisine distributions (for cuisine CSV)
# =============================================================
def load_cuisine_counts(
    enriched_root: Path, providers: list[str],
    variant_filter: Optional[str],
) -> dict[tuple, Counter]:
    """For each (provider, category), accumulate cuisine tag counts."""
    out: dict[tuple, Counter] = defaultdict(Counter)
    for provider in providers:
        d = enriched_root / provider
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if "_skipped" in (rec.get("grounding") or {}):
                continue
            if variant_filter and rec.get("variant") != variant_filter:
                continue
            wd = (rec.get("grounding") or {}).get("wikidata") or {}
            dist = wd.get("cuisine_distribution") or {}
            key = (provider, rec.get("category"))
            for cuisine, count in dist.items():
                out[key][cuisine] += int(count)
    return out


# =============================================================
# CSV writers
# =============================================================
def _avg(lst: list[float]) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def write_per_response_csv(rows: list[dict], path: Path):
    if not rows:
        print(f"  [skip] no rows; not writing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_summary_csv(agg: dict[tuple, CategoryAggregate], path: Path):
    rows = []
    for (provider, category), a in sorted(agg.items()):
        rows.append({
            "provider": provider,
            "category": category,
            "n_responses": a.n,
            "wikidata_coverage_avg": round(_avg(a.wd_coverage), 3),
            "bls_coverage_avg": round(_avg(a.bls_coverage), 3),
            "compendium_coverage_avg_fitness_only": (
                round(_avg(a.comp_coverage), 3) if a.comp_coverage else None
            ),
            "fitness_response_count": len(a.comp_coverage),
            "western_centricity_avg": (
                round(_avg(a.western_centricity), 3)
                if a.western_centricity else None
            ),
            "western_centricity_responses_counted": len(a.western_centricity),
        })
    if not rows:
        print(f"  [skip] no aggregates; not writing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_class_distribution_csv(
    agg: dict[tuple, CategoryAggregate], path: Path,
    attr: str, value_col: str,
):
    """Long-format CSV: one row per (provider, category, class) with count."""
    rows = []
    for (provider, category), a in sorted(agg.items()):
        counter: Counter = getattr(a, attr)
        total = sum(counter.values())
        for cls, count in counter.most_common():
            rows.append({
                "provider": provider,
                "category": category,
                value_col: cls,
                "count": count,
                "share": round(count / total, 3) if total else 0.0,
            })
    if not rows:
        print(f"  [skip] no entries; not writing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_cuisine_distribution_csv(
    cuisine_counts: dict[tuple, Counter], path: Path, top_k: int = 15,
):
    """Top-k cuisines per (provider, category)."""
    rows = []
    for (provider, category), counter in sorted(cuisine_counts.items()):
        total = sum(counter.values())
        for cuisine, count in counter.most_common(top_k):
            rows.append({
                "provider": provider,
                "category": category,
                "cuisine": cuisine,
                "count": count,
                "share": round(count / total, 3) if total else 0.0,
            })
    if not rows:
        print(f"  [skip] no cuisines; not writing {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


# =============================================================
# Console summary
# =============================================================
def print_console_summary(rows: list[dict], agg: dict[tuple, CategoryAggregate]):
    print("=" * 72)
    print("PHASE 4 COVERAGE REPORT")
    print("=" * 72)
    print(f"Total responses analyzed: {len(rows)}")

    print("\nPer-source coverage averaged across applicable responses:")
    fitness_rows = [r for r in rows if r["has_fitness_content"]]
    cents = [r["western_centricity"] for r in rows
             if r["western_centricity"] is not None]

    print(f"  Wikidata (cuisine):   {_avg([r['wd_coverage']  for r in rows]):.1%}  "
          f"({len(rows)} responses)")
    print(f"  BLS prices:           {_avg([r['bls_coverage'] for r in rows]):.1%}  "
          f"({len(rows)} responses)")
    print(f"  Compendium (fitness): {_avg([r['comp_coverage'] for r in fitness_rows]):.1%}  "
          f"({len(fitness_rows)} fitness-bearing responses)")
    if cents:
        print(f"  Western centricity:   {_avg(cents):.1%}  "
              f"({len(cents)} responses with cuisine tags)")

    cost_status_dist = Counter(r["cost_period_status"] for r in rows)
    print("\nCost-period status distribution:")
    for status, n in cost_status_dist.most_common():
        print(f"  {status:24s} {n}")

    print("\nBy provider x category:")
    print(f"  {'provider':10} {'category':10} {'n':>4} "
          f"{'wd':>6} {'bls':>6} {'comp':>6} {'west%':>6}")
    for (provider, category), a in sorted(agg.items()):
        west = (
            f"{_avg(a.western_centricity):6.1%}"
            if a.western_centricity else "   n/a"
        )
        comp = (
            f"{_avg(a.comp_coverage):6.1%}"
            if a.comp_coverage else "   n/a"
        )
        print(f"  {provider:10} {category:10} {a.n:4d} "
              f"{_avg(a.wd_coverage):6.1%} "
              f"{_avg(a.bls_coverage):6.1%} "
              f"{comp} {west}")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 4 coverage report")
    parser.add_argument("--in-dir", default=ENRICHED_DIR,
                        help="Input directory of enriched files.")
    parser.add_argument("--out-dir", default=RESULTS_DIR,
                        help="Output directory for CSVs.")
    parser.add_argument("--provider", default=None,
                        choices=list(MODELS.keys()),
                        help="Restrict report to one provider.")
    parser.add_argument("--variant", default=None,
                        choices=["baseline", "constrained"],
                        help="Restrict report to baseline or constrained.")
    args = parser.parse_args()

    enriched_root = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not enriched_root.exists():
        print(f"No enriched directory at {enriched_root}. "
              "Run `python -m src.ground_all` first.")
        return

    if args.provider:
        providers = [args.provider]
    else:
        providers = list(MODELS.keys())

    skipped: Counter = Counter()
    rows = collect_rows(enriched_root, providers, args.variant, skipped)

    if skipped:
        print("Skipped during load:")
        for reason, n in skipped.items():
            print(f"  {reason}: {n}")

    agg = build_aggregates(rows)
    cuisine_counts = load_cuisine_counts(enriched_root, providers, args.variant)

    print()
    write_per_response_csv(rows, out_dir / "coverage_report.csv")
    write_summary_csv(agg, out_dir / "coverage_summary.csv")
    write_class_distribution_csv(
        agg, out_dir / "thrifty_classification.csv",
        attr="thrifty_classes", value_col="classification",
    )
    write_class_distribution_csv(
        agg, out_dir / "feasibility_assessment.csv",
        attr="feasibility_classes", value_col="feasibility_class",
    )
    write_cuisine_distribution_csv(
        cuisine_counts, out_dir / "cuisine_distribution.csv",
    )

    print()
    print_console_summary(rows, agg)
    print()
    print("Use coverage_summary.csv as the basis for Table 1 in the paper.")


if __name__ == "__main__":
    main()
