"""
src/aggregate_scores.py

Phase 5, Track E — Build the master scores.csv.

Joins:
  - Judge scores from data/judged/{provider}/{prompt_id}.json
  - Similarity signals from results/similarity.csv
  - Grounding metrics carried in the judged record's "grounding" block
  - Arena win rates from results/arena_matrix.csv (per provider × dimension)

Output:
  results/scores.csv               — one row per (provider, prompt_id), all signals
  results/scores_summary.csv       — provider × category × variant aggregates
  results/scores_by_provider.csv   — overall per-provider mean scores
  results/scores_adherence_branches.csv — per-branch adherence yes/partial/no shares

scores.csv is the master table feeding Phase 6 (validation) and Phase 7
(figures, paper).

Run:
    python -m src.aggregate_scores
"""

import argparse
import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    MODELS, MODEL_DISPLAY_NAMES, JUDGED_DIR, RESULTS_DIR,
)


def _safe(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# =============================================================
# Loaders
# =============================================================
def load_similarity(path: Path) -> dict[tuple, dict]:
    """Both members of a prompt pair carry the same similarity signal
    (it's a pairwise metric). Indexed by both baseline_id and
    constrained_id so any record can look up its pair."""
    if not path.exists():
        return {}
    out: dict[tuple, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sig = {
                "cosine_full":         float(row.get("cosine_full") or 0),
                "cosine_ingredients":  float(row.get("cosine_ingredients") or 0),
                "cosine_structural":   float(row.get("cosine_structural") or 0),
                "jaccard_ingredients": float(row.get("jaccard_ingredients") or 0),
                "pair_key":            row.get("pair_key"),
            }
            out[(row["provider"], row["baseline_id"])] = sig
            out[(row["provider"], row["constrained_id"])] = sig
    return out


def load_arena_winrates(path: Path) -> dict[tuple, dict]:
    """Aggregate arena_matrix.csv into per-(provider, dimension) win rates.
    A model's 'decided' pool aggregates its wins+losses across all pair cells
    it appeared in. Win rate excludes ties."""
    if not path.exists():
        return {}
    by_provider: dict[tuple, dict] = defaultdict(
        lambda: {"wins": 0, "decided": 0, "ties": 0}
    )
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            x = row["provider_x"]
            y = row["provider_y"]
            dim = row["dimension"]
            n_x_wins = int(row["n_x_wins"])
            n_y_wins = int(row["n_y_wins"])
            n_ties = int(row["n_ties"])

            by_provider[(x, dim)]["wins"] += n_x_wins
            by_provider[(x, dim)]["decided"] += n_x_wins + n_y_wins
            by_provider[(x, dim)]["ties"] += n_ties

            by_provider[(y, dim)]["wins"] += n_y_wins
            by_provider[(y, dim)]["decided"] += n_x_wins + n_y_wins
            by_provider[(y, dim)]["ties"] += n_ties

    out = {}
    for key, d in by_provider.items():
        if d["decided"] > 0:
            out[key] = {
                "winrate": round(d["wins"] / d["decided"], 3),
                "wins": d["wins"],
                "decided": d["decided"],
                "ties": d["ties"],
            }
    return out


# =============================================================
# Per-record row construction
# =============================================================
def build_rows(providers: list[str]) -> list[dict]:
    similarity = load_similarity(Path(RESULTS_DIR) / "similarity.csv")

    rows: list[dict] = []
    for provider in providers:
        provider_dir = Path(JUDGED_DIR) / provider
        if not provider_dir.exists():
            print(f"  no judged dir for {provider}, skipping")
            continue

        for f in sorted(provider_dir.glob("*.json")):
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            judges = rec.get("judges") or {}
            if "_skipped" in judges:
                continue

            prompt_id = rec.get("prompt_id")
            sim = similarity.get((provider, prompt_id), {})

            grounding = rec.get("grounding") or {}
            wd = grounding.get("wikidata") or {}
            bls = grounding.get("bls_prices") or {}
            thrifty = grounding.get("thrifty_plan") or {}
            comp = grounding.get("compendium") or {}

            row = {
                # Identity
                "provider": provider,
                "model_display": MODEL_DISPLAY_NAMES.get(provider, provider),
                "prompt_id": prompt_id,
                "category": rec.get("category"),
                "variant": rec.get("variant"),
                "category_type": rec.get("category_type") or "",

                # Judge scores
                "score_affordability": _safe(judges, "affordability", "score"),
                "score_cultural":      _safe(judges, "cultural", "score"),
                "score_feasibility":   _safe(judges, "feasibility", "score"),
                "score_adherence":     _safe(judges, "adherence", "score"),
                "adherence_applicable_branches":
                    _safe(judges, "adherence", "applicable_branches", default=0),

                # Adherence per-branch verdicts (string "yes"/"partial"/"no"/"not_applicable")
                "adherence_b1_financial":
                    _safe(judges, "adherence", "branch_verdicts",
                          "branch_1_financial", "verdict"),
                "adherence_b2_cultural":
                    _safe(judges, "adherence", "branch_verdicts",
                          "branch_2_cultural", "verdict"),
                "adherence_b3_lifestyle":
                    _safe(judges, "adherence", "branch_verdicts",
                          "branch_3_lifestyle", "verdict"),

                # Similarity (paired with the variant counterpart)
                "cosine_full":         sim.get("cosine_full"),
                "cosine_ingredients":  sim.get("cosine_ingredients"),
                "cosine_structural":   sim.get("cosine_structural"),
                "jaccard_ingredients": sim.get("jaccard_ingredients"),

                # Grounding signals
                "wd_coverage":          wd.get("coverage_ratio"),
                "western_centricity":   wd.get("western_centricity_ratio"),
                "bls_coverage":         bls.get("coverage_ratio"),
                "thrifty_classification": thrifty.get("classification"),
                "thrifty_ratio":        thrifty.get("response_cost_vs_thrifty_ratio"),
                "feasibility_assessment": comp.get("feasibility_assessment"),
                "comp_coverage":        comp.get("coverage_ratio"),
                "moderate_equiv_minutes": comp.get("moderate_equivalent_minutes"),

                # Parse-error flags (paper transparency)
                "affordability_parse_error":
                    bool(_safe(judges, "affordability", "parse_error")),
                "cultural_parse_error":
                    bool(_safe(judges, "cultural", "parse_error")),
                "feasibility_parse_error":
                    bool(_safe(judges, "feasibility", "parse_error")),
                "adherence_parse_error":
                    bool(_safe(judges, "adherence", "parse_error")),
            }
            rows.append(row)
    return rows


# =============================================================
# Aggregates
# =============================================================
def _mean(lst: list) -> Optional[float]:
    vals = [v for v in lst if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _count_non_null(lst: list) -> int:
    return sum(1 for v in lst if v is not None)


def build_summary(rows: list[dict]) -> list[dict]:
    """Provider × category × variant aggregates with per-metric N counts."""
    SIGS = ["score_affordability", "score_cultural",
            "score_feasibility", "score_adherence",
            "cosine_full", "cosine_ingredients",
            "cosine_structural", "jaccard_ingredients",
            "western_centricity"]
    bucket: dict[tuple, dict] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["provider"], row["category"], row["variant"])
        for sig in SIGS:
            bucket[key][sig].append(row.get(sig))

    out = []
    for (prov, cat, var), sigs in sorted(bucket.items()):
        record = {
            "provider": prov,
            "category": cat,
            "variant":  var,
            "n_total":  len(sigs["score_affordability"]),
        }
        for sig in SIGS:
            record[f"{sig}_mean"] = _mean(sigs[sig])
            record[f"{sig}_n"] = _count_non_null(sigs[sig])
        out.append(record)
    return out


def build_provider_summary(
    rows: list[dict], arena: dict[tuple, dict],
) -> list[dict]:
    """Per-provider overall summary plus arena win rates per dimension."""
    bucket: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        prov = row["provider"]
        for sig in ["score_affordability", "score_cultural",
                    "score_feasibility", "score_adherence"]:
            bucket[prov][sig].append(row.get(sig))

    out = []
    for prov, sigs in sorted(bucket.items()):
        record = {
            "provider": prov,
            "model_display": MODEL_DISPLAY_NAMES.get(prov, prov),
            "n_total": len(sigs["score_affordability"]),
        }
        for sig in ["score_affordability", "score_cultural",
                    "score_feasibility", "score_adherence"]:
            record[f"{sig}_mean"] = _mean(sigs[sig])
            record[f"{sig}_n"] = _count_non_null(sigs[sig])
        for dim in ("affordability", "cultural", "feasibility"):
            arena_data = arena.get((prov, dim))
            if arena_data:
                record[f"arena_{dim}_winrate"] = arena_data["winrate"]
                record[f"arena_{dim}_decided"] = arena_data["decided"]
                record[f"arena_{dim}_ties"] = arena_data["ties"]
            else:
                record[f"arena_{dim}_winrate"] = None
                record[f"arena_{dim}_decided"] = 0
                record[f"arena_{dim}_ties"] = 0
        out.append(record)
    return out


def build_adherence_branch_summary(rows: list[dict]) -> list[dict]:
    """Per-(provider, category, variant, branch) yes/partial/no/NA share.
    These are the headline RQ adherence numbers for the paper."""
    BRANCHES = {
        "branch_1_financial":  "adherence_b1_financial",
        "branch_2_cultural":   "adherence_b2_cultural",
        "branch_3_lifestyle":  "adherence_b3_lifestyle",
    }

    bucket: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        for branch_label, col in BRANCHES.items():
            v = row.get(col)
            key = (row["provider"], row["category"], row["variant"], branch_label)
            bucket[key][v if v is not None else "missing"] += 1

    out = []
    for (prov, cat, var, branch), counts in sorted(bucket.items()):
        total = sum(counts.values())
        applicable = total - counts.get("not_applicable", 0) - counts.get("missing", 0)
        out.append({
            "provider": prov,
            "category": cat,
            "variant":  var,
            "branch":   branch,
            "n_total":  total,
            "n_applicable": applicable,
            "n_yes":     counts.get("yes", 0),
            "n_partial": counts.get("partial", 0),
            "n_no":      counts.get("no", 0),
            "n_not_applicable": counts.get("not_applicable", 0),
            "n_missing": counts.get("missing", 0),
            "share_yes_among_applicable": (
                round(counts.get("yes", 0) / applicable, 3)
                if applicable > 0 else None
            ),
            "share_no_among_applicable": (
                round(counts.get("no", 0) / applicable, 3)
                if applicable > 0 else None
            ),
        })
    return out


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase 5 scores")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    providers = list(MODELS.keys())

    judged_root = Path(JUDGED_DIR)
    if not judged_root.exists():
        print(f"ERROR: {judged_root} does not exist. "
              f"Run `python -m src.run_judges` first.")
        return

    print("Loading judged records and similarity...")
    rows = build_rows(providers)
    print(f"  {len(rows)} per-response rows assembled")

    if not rows:
        print("No rows to aggregate. Did Phase 5 run_judges complete?")
        return

    print("Loading arena results...")
    arena = load_arena_winrates(Path(RESULTS_DIR) / "arena_matrix.csv")
    print(f"  {len(arena)} (provider, dimension) arena cells loaded")

    summary = build_summary(rows)
    provider_summary = build_provider_summary(rows, arena)
    branch_summary = build_adherence_branch_summary(rows)

    # Write outputs
    def _write(rows_list, path):
        if not rows_list:
            print(f"  [skip] no rows for {path}")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_list[0].keys()))
            writer.writeheader()
            writer.writerows(rows_list)
        print(f"  wrote {path} ({len(rows_list)} rows)")

    _write(rows, out_dir / "scores.csv")
    _write(summary, out_dir / "scores_summary.csv")
    _write(provider_summary, out_dir / "scores_by_provider.csv")
    _write(branch_summary, out_dir / "scores_adherence_branches.csv")

    # Console summary
    def _fmt(x, width: int, fmt: str) -> str:
        return f"{x:{fmt}}" if x is not None else f"{'n/a':>{width}}"

    print("\n" + "=" * 90)
    print("PHASE 5 AGGREGATE SUMMARY")
    print("=" * 90)
    print(f"Total scored responses: {len(rows)}")

    print("\nMean judge scores by provider (1=worst, 5=best):")
    print(f"  {'provider':10} {'aff':>8} {'cult':>8} {'feas':>8} {'adh':>8} "
          f"{'arena_aff':>11} {'arena_cult':>11} {'arena_feas':>11}")
    for ps in provider_summary:
        print(f"  {ps['provider']:10} "
              f"{_fmt(ps['score_affordability_mean'], 8, '8.2f')} "
              f"{_fmt(ps['score_cultural_mean'],      8, '8.2f')} "
              f"{_fmt(ps['score_feasibility_mean'],   8, '8.2f')} "
              f"{_fmt(ps['score_adherence_mean'],     8, '8.2f')} "
              f"{_fmt(ps.get('arena_affordability_winrate'), 11, '11.1%')} "
              f"{_fmt(ps.get('arena_cultural_winrate'),       11, '11.1%')} "
              f"{_fmt(ps.get('arena_feasibility_winrate'),    11, '11.1%')}")

    print("\nNext: Phase 6 (human validation, Cohen's kappa)")


if __name__ == "__main__":
    main()
