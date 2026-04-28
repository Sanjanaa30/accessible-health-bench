"""
src/ground_all.py

Phase 4 orchestrator.

Runs all four grounding passes on every extraction file in data/extractions/
and writes enriched output to data/enriched/.

  4a. Wikidata cuisine grounding   (RQ2 — cultural appropriateness)
  4b. BLS retail food prices       (RQ1 — affordability, per-ingredient)
  4c. USDA Cost of Food at Home    (RQ1 — affordability calibration)
  4d. Compendium activities        (RQ3 — fitness feasibility)

Each enriched file is the original extraction record plus a top-level
`grounding` block consumed by the Phase 5 judges, plus a `_grounding_meta`
block recording snapshot dates and SHA256s for paper-quality reproducibility.

Three-pass design:
  Pass 1: ground every extraction (Wikidata accumulates misses in-process).
  Pass 2: one batched LLM-fallback call resolves all Wikidata misses.
  Pass 3: finalize each in-memory result with the fallback cuisines and write.

Run from repo root:
    python -m src.ground_all                  # full run, restart-safe
    python -m src.ground_all --pilot 5        # 5 extractions per provider
    python -m src.ground_all --providers openai
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import MODELS, EXTRACTIONS_DIR, ENRICHED_DIR
from src.grounding.wikidata import WikidataGrounder, normalize_food_name
from src.grounding.bls import BLSGrounder
from src.grounding.thrifty_plan import ThriftyPlanGrounder
from src.grounding.compendium import CompendiumGrounder

# Cuisines the dashboard / paper treats as Western for the centricity metric.
# Documented criterion: Western European + Anglo-American + Mediterranean
# (the last contested but conventional in nutrition lit). North African
# cuisines are deliberately NOT included even when sometimes Mediterranean-
# tagged, because the metric is meant to surface Anglo-Mediterranean default
# bias.
WESTERN_CUISINE_TAGS = {
    "western", "american", "european",
    "italian", "french", "spanish", "greek",
    "british", "irish", "german",
    "nordic", "scandinavian", "portuguese",
    "mediterranean",
}

# How many cost periods we know how to normalize to per-week.
COST_PERIOD_TO_WEEKLY = {
    "per_week": 1.0,
    "per_day": 7.0,
    "per_month": 12.0 / 52.0,
    # per_meal is extrapolated using a 3-meals/day x 7-days = 21 multiplier.
    # The status flag "per_meal_extrapolated" preserves the assumption in
    # the per-response record so the paper can disclose it.
    "per_meal": 21.0,
}


@dataclass
class EnrichmentTask:
    provider: str
    prompt_id: str
    source_record: dict
    grounding: dict


# =============================================================
# Per-block grounders
# =============================================================
def ground_wikidata(extracted: dict, grounder: WikidataGrounder) -> dict:
    """Wikidata pass 1: lookups + miss accumulation. Finalized in pass 3."""
    ingredients = extracted.get("all_ingredients", []) or []
    dishes = extracted.get("all_dishes_or_foods_named", []) or []

    all_foods = sorted({
        normalize_food_name(f) for f in (ingredients + dishes) if f
    })

    if not all_foods:
        return {
            "ingredients_grounded": [],
            "matched_count": 0,
            "total_count": 0,
            "coverage_ratio": 0.0,
            "cuisine_distribution": {},
            "western_centricity_ratio": None,
        }

    results = grounder.lookup_batch(all_foods)
    return {
        "ingredients_grounded": results,
        "matched_count": sum(1 for r in results if r.get("cuisines")),
        "total_count": len(all_foods),
    }


def finalize_wikidata(grounding_block: dict, fallback_results: dict) -> dict:
    """Pass 3: merge LLM-fallback cuisines and compute distribution metrics."""
    grounded = grounding_block.get("ingredients_grounded", [])
    if not grounded:
        return {
            "ingredients_grounded": [],
            "matched_count": 0,
            "total_count": 0,
            "coverage_ratio": 0.0,
            "cuisine_distribution": {},
            "western_centricity_ratio": None,
        }

    # Build a new list (don't mutate cached dicts) merging fallback hits.
    merged: list[dict] = []
    for entry in grounded:
        # entry["food_name"] is already normalized — don't re-normalize.
        key = entry.get("food_name")
        if not entry.get("cuisines") and key in fallback_results:
            fb = fallback_results[key]
            merged.append({
                **entry,
                "cuisines": fb.get("cuisines", []),
                "source": fb.get("source", entry.get("source")),
                "confidence": fb.get("confidence", entry.get("confidence")),
            })
        else:
            merged.append(entry)

    cuisine_counts: Counter = Counter()
    for entry in merged:
        for c in entry.get("cuisines", []):
            cuisine_counts[c] += 1

    total_tags = sum(cuisine_counts.values())
    western_tags = sum(
        cuisine_counts.get(c, 0) for c in WESTERN_CUISINE_TAGS
    )
    matched = sum(1 for e in merged if e.get("cuisines"))

    return {
        "ingredients_grounded": merged,
        "matched_count": matched,
        "total_count": len(merged),
        "coverage_ratio": matched / len(merged) if merged else 0.0,
        "cuisine_distribution": dict(cuisine_counts.most_common()),
        "western_centricity_ratio": (
            round(western_tags / total_tags, 3) if total_tags > 0 else None
        ),
        "western_tag_count": western_tags,
        "total_tag_count": total_tags,
    }


def ground_bls(extracted: dict, grounder: BLSGrounder) -> dict:
    """BLS per-ingredient unit prices + coverage report."""
    ingredients = extracted.get("all_ingredients", []) or []
    if not ingredients:
        return {
            "matched": [],
            "unmatched": [],
            "matched_count": 0,
            "total_unique_ingredients": 0,
            "coverage_ratio": 0.0,
            "snapshot": grounder.manifest_info(),
        }
    return grounder.coverage_report(ingredients)


def _normalize_response_cost_to_weekly(extracted: dict) -> tuple[Optional[float], str]:
    """
    Pull the response's own cost estimate from extraction and normalize to
    per-week USD. Returns (weekly_cost_or_None, status_string).

    Status values: 'normalized', 'no_cost', 'period_unknown', 'period_too_vague'.
    """
    cost_info = extracted.get("cost_information", {}) or {}
    raw_cost = cost_info.get("total_cost_usd")
    period = cost_info.get("cost_period")

    if raw_cost in (None, 0, 0.0):
        return None, "no_cost"

    try:
        c = float(raw_cost)
    except (TypeError, ValueError):
        return None, "no_cost"

    if c <= 0:
        return None, "no_cost"

    if period not in COST_PERIOD_TO_WEEKLY:
        return None, "period_unknown"

    weekly = c * COST_PERIOD_TO_WEEKLY[period]
    status = "per_meal_extrapolated" if period == "per_meal" else "normalized"
    return round(weekly, 2), status


def ground_thrifty(extracted: dict, grounder: ThriftyPlanGrounder) -> dict:
    """USDA Thrifty Plan calibration of the response's own cost claim."""
    household = extracted.get("household_and_demographic_context", {}) or {}
    size = household.get("household_size_implied")
    htype = household.get("household_type")
    ages = household.get("ages_referenced") or None

    weekly_cost, cost_status = _normalize_response_cost_to_weekly(extracted)
    result = grounder.classify_household_response(
        response_estimated_cost_usd=weekly_cost,
        household_size=size,
        household_type=htype,
        ages_referenced=ages,
    )

    baseline = result.get("baseline", {})
    weekly_baseline = baseline.get("weekly_baseline_usd", {}) or {}
    thrifty_value = weekly_baseline.get("thrifty")
    if weekly_cost is not None and thrifty_value:
        ratio = round(weekly_cost / thrifty_value, 3)
    else:
        ratio = None

    return {
        **baseline,
        "response_estimated_cost_weekly_usd": weekly_cost,
        "response_cost_period_status": cost_status,
        "response_cost_vs_thrifty_ratio": ratio,
        "classification": result.get("bucket"),
    }


def ground_compendium(extracted: dict, grounder: CompendiumGrounder) -> dict:
    """Compendium MET grounding + WHO bucketing (only when time_horizon=weekly)."""
    fitness = extracted.get("fitness_components", []) or []
    routine = extracted.get("routine_structure", {}) or {}
    time_horizon = routine.get("time_horizon")

    if not fitness:
        return grounder._empty_coverage(time_horizon)
    return grounder.coverage_report(fitness, time_horizon=time_horizon)


# =============================================================
# Path helpers
# =============================================================
def extraction_path(provider: str, prompt_id: str) -> Path:
    return Path(EXTRACTIONS_DIR) / provider / f"{prompt_id}.json"


def enriched_path(provider: str, prompt_id: str) -> Path:
    return Path(ENRICHED_DIR) / provider / f"{prompt_id}.json"


def already_enriched(provider: str, prompt_id: str) -> bool:
    return enriched_path(provider, prompt_id).exists()


def save_enriched(task: EnrichmentTask, grounding_meta: dict) -> Path:
    out_path = enriched_path(task.provider, task.prompt_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(task.source_record)
    record["grounding"] = task.grounding
    record["_grounding_meta"] = grounding_meta
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return out_path


# =============================================================
# Main orchestrator
# =============================================================
def ground_all(
    pilot: Optional[int] = None,
    providers: Optional[list[str]] = None,
):
    if providers is None:
        providers = list(MODELS.keys())

    print("Initializing grounders...")
    wd = WikidataGrounder()
    bls = BLSGrounder()
    thrifty = ThriftyPlanGrounder()
    comp = CompendiumGrounder()

    if not bls.entries:
        print("  WARNING: BLS data not loaded. "
              "Run download_external_data.py first.")
    if not thrifty.table:
        print("  WARNING: Thrifty Plan data not loaded.")
    if not comp.entries:
        print("  WARNING: Compendium data not loaded.")

    grounding_meta_template = {
        "wikidata": {"cache_stats": wd.cache_stats()},
        "bls": bls.manifest_info(),
        "thrifty_plan": thrifty.manifest_info(),
        "compendium": comp.manifest_info(),
    }

    # Build task list
    tasks: list[tuple[str, Path]] = []
    for provider in providers:
        provider_dir = Path(EXTRACTIONS_DIR) / provider
        if not provider_dir.exists():
            print(f"  no extractions for {provider} — skipping")
            continue
        ext_files = sorted(provider_dir.glob("*.json"))
        if pilot is not None:
            ext_files = ext_files[:pilot]
        for ef in ext_files:
            tasks.append((provider, ef))

    print(f"\nPlan: {len(tasks)} extractions to ground\n")

    # ---------------------------------------------------------
    # Pass 1: ground every extraction (Wikidata accumulates misses)
    # ---------------------------------------------------------
    interim: list[EnrichmentTask] = []
    skipped_already = 0
    skipped_extraction_error = 0

    with tqdm(total=len(tasks), desc="Grounding (pass 1)", unit="resp") as pbar:
        for provider, ext_file in tasks:
            source_record = json.loads(ext_file.read_text(encoding="utf-8"))
            prompt_id = source_record["prompt_id"]

            if already_enriched(provider, prompt_id):
                skipped_already += 1
                pbar.update(1)
                pbar.set_postfix_str(f"skipped {provider}/{prompt_id}")
                continue

            extracted = source_record.get("extracted", {}) or {}
            if isinstance(extracted, dict) and "_extraction_error" in extracted:
                grounding = {"_skipped": "extraction_error"}
                save_enriched(
                    EnrichmentTask(provider, prompt_id, source_record, grounding),
                    grounding_meta_template,
                )
                skipped_extraction_error += 1
                pbar.update(1)
                continue

            grounding = {
                "wikidata":     ground_wikidata(extracted, wd),
                "bls_prices":   ground_bls(extracted, bls),
                "thrifty_plan": ground_thrifty(extracted, thrifty),
                "compendium":   ground_compendium(extracted, comp),
            }
            interim.append(
                EnrichmentTask(provider, prompt_id, source_record, grounding)
            )
            pbar.update(1)
            pbar.set_postfix_str(f"{provider}/{prompt_id}")

    # ---------------------------------------------------------
    # Pass 2: single batched LLM call resolves all Wikidata misses
    # ---------------------------------------------------------
    print(f"\nResolving Wikidata misses via LLM batch...")
    fallback_results = wd.resolve_misses()
    print(f"  resolved {len(fallback_results)} miss(es)")

    # ---------------------------------------------------------
    # Pass 3: finalize Wikidata + write enriched files
    # ---------------------------------------------------------
    print("\nWriting enriched files...")
    for task in tqdm(interim, desc="Finalizing", unit="file"):
        task.grounding["wikidata"] = finalize_wikidata(
            task.grounding["wikidata"], fallback_results
        )
        save_enriched(task, grounding_meta_template)

    # ---------------------------------------------------------
    # Summary + persisted aggregate
    # ---------------------------------------------------------
    summary = _summarize(providers)
    print("\n" + "=" * 60)
    print("GROUNDING COMPLETE")
    print("=" * 60)
    print(f"Skipped (already enriched):    {skipped_already}")
    print(f"Skipped (extraction error):    {skipped_extraction_error}")
    print(f"Wikidata fallbacks resolved:   {len(fallback_results)}")
    print()
    print("Files per provider:")
    for prov, count in summary["files_per_provider"].items():
        print(f"  {prov:12s} {count}")

    print()
    print("Coverage (averaged across enriched files with applicable content):")
    for k, v in summary["coverage"].items():
        if v["responses_counted"]:
            print(f"  {k:14s} {v['mean']:.1%}  "
                  f"({v['responses_counted']} responses)")
        else:
            print(f"  {k:14s} -- no responses counted")

    summary_path = Path(ENRICHED_DIR) / "_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["grounding_meta"] = grounding_meta_template
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSummary written to {summary_path}")
    print("Ready for Phase 5 (judging).")


def _summarize(providers: list[str]) -> dict:
    """Aggregate coverage across enriched files."""
    files_per_provider: dict[str, int] = {}
    wd_cov, bls_cov, comp_cov = [], [], []
    western_ratios = []
    cost_classes: Counter = Counter()
    feasibility_classes: Counter = Counter()

    for provider in providers:
        d = Path(ENRICHED_DIR) / provider
        files_per_provider[provider] = (
            len(list(d.glob("*.json"))) if d.exists() else 0
        )
        if not d.exists():
            continue

        for f in d.glob("*.json"):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            g = rec.get("grounding", {}) or {}
            if "_skipped" in g:
                continue

            wd_block = g.get("wikidata", {})
            if wd_block.get("total_count", 0) > 0:
                wd_cov.append(wd_block.get("coverage_ratio", 0.0))
                if wd_block.get("western_centricity_ratio") is not None:
                    western_ratios.append(wd_block["western_centricity_ratio"])

            bls_block = g.get("bls_prices", {})
            if bls_block.get("total_unique_ingredients", 0) > 0:
                bls_cov.append(bls_block.get("coverage_ratio", 0.0))

            comp_block = g.get("compendium", {})
            if comp_block.get("total_count", 0) > 0:
                comp_cov.append(comp_block.get("coverage_ratio", 0.0))
                fb = comp_block.get("feasibility_assessment")
                if fb:
                    feasibility_classes[fb] += 1

            thrifty_block = g.get("thrifty_plan", {})
            cls = thrifty_block.get("classification")
            if cls:
                cost_classes[cls] += 1

    def _stats(lst):
        return {
            "responses_counted": len(lst),
            "mean": sum(lst) / len(lst) if lst else 0.0,
        }

    return {
        "files_per_provider": files_per_provider,
        "coverage": {
            "wikidata":   _stats(wd_cov),
            "bls":        _stats(bls_cov),
            "compendium": _stats(comp_cov),
        },
        "western_centricity": {
            "responses_counted": len(western_ratios),
            "mean": (
                round(sum(western_ratios) / len(western_ratios), 3)
                if western_ratios else None
            ),
        },
        "thrifty_classification_distribution": dict(cost_classes.most_common()),
        "compendium_feasibility_distribution": dict(
            feasibility_classes.most_common()
        ),
    }


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ground extracted responses against Wikidata, BLS, USDA, "
                    "and the 2024 Compendium.",
    )
    parser.add_argument("--pilot", type=int, default=None,
                        help="Only ground first N extractions per provider.")
    parser.add_argument(
        "--providers", nargs="+", default=None,
        choices=["openai", "anthropic", "deepseek", "groq"],
    )
    args = parser.parse_args()

    ground_all(pilot=args.pilot, providers=args.providers)
