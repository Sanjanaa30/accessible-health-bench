"""
src/compute_kappa.py

Phase 6, Step 2 — Cohen's kappa validation.

Reads the two filled-in human-rater CSVs (from src/sample_validation_set.py)
and the LLM judge scores (from data/judged/), and computes:

  1. Inter-human kappa per dimension
       — how much do the two humans agree with each other?
       — this is the upper bound on what any LLM judge can achieve.

  2. Judge vs human-consensus kappa per dimension
       — for each LLM judge, how much does it agree with the human
         consensus (defined as: both humans agreed exactly).

  3. DAG branch agreement (% of branches where verdict matches)
       — for the adherence judge, three branches (financial / cultural /
         lifestyle) are compared individually.

Output:
  results/kappa_report.csv          — one row per (rater_pair, dimension)
  console summary

Run from repo root:
    python -m src.compute_kappa
    python -m src.compute_kappa --rater-names sanjana snigdha
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import JUDGED_DIR, RESULTS_DIR


VALIDATION_DIR = Path(RESULTS_DIR) / "validation"
DIMENSIONS = ["affordability", "cultural", "feasibility", "health_accuracy"]
DAG_BRANCHES = {
    "branch_1_financial":  "human_adherence_b1_financial",
    "branch_2_cultural":   "human_adherence_b2_cultural",
    "branch_3_lifestyle":  "human_adherence_b3_lifestyle",
}
VALID_VERDICTS = {"yes", "partial", "no", "not_applicable"}


# =============================================================
# Cohen's kappa from scratch (no scikit-learn dependency for this one
# function; we use sklearn elsewhere for the ML baseline).
# =============================================================
def cohens_kappa(rater_a: list, rater_b: list) -> Optional[float]:
    """Compute Cohen's kappa for two equal-length sequences of categorical
    labels. Pairs where either side is None are dropped.

    Returns None if fewer than 2 valid pairs remain or if all labels are
    identical (kappa is undefined when one rater never varies)."""
    pairs = [(a, b) for a, b in zip(rater_a, rater_b)
             if a is not None and b is not None]
    if len(pairs) < 2:
        return None

    labels = sorted({lbl for pair in pairs for lbl in pair})
    if len(labels) < 2:
        # Only one label seen across both raters — kappa undefined.
        return None

    n = len(pairs)
    # Observed agreement
    p_obs = sum(1 for a, b in pairs if a == b) / n

    # Expected agreement by chance
    counts_a = Counter(a for a, _ in pairs)
    counts_b = Counter(b for _, b in pairs)
    p_exp = sum(
        (counts_a.get(lbl, 0) / n) * (counts_b.get(lbl, 0) / n)
        for lbl in labels
    )

    if abs(1 - p_exp) < 1e-9:
        return None  # avoid division by zero
    return (p_obs - p_exp) / (1 - p_exp)


def kappa_band(k: Optional[float]) -> str:
    """Landis & Koch (1977) interpretation, the standard reference."""
    if k is None:
        return "undefined"
    if k < 0:           return "less than chance"
    if k < 0.20:        return "slight"
    if k < 0.40:        return "fair"
    if k < 0.60:        return "moderate"
    if k < 0.80:        return "substantial"
    return "almost perfect"


# =============================================================
# Loading
# =============================================================
def _to_int_or_none(s: str) -> Optional[int]:
    """Parse a 1-5 cell into an int; return None for blank or invalid."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        v = int(round(float(s)))
    except (TypeError, ValueError):
        return None
    if 1 <= v <= 5:
        return v
    return None


def _verdict_or_none(s: str) -> Optional[str]:
    """Normalize an adherence verdict to one of yes/partial/no/not_applicable."""
    if s is None:
        return None
    s = str(s).strip().lower().replace(" ", "_")
    return s if s in VALID_VERDICTS else None


def _read_csv_tolerant(path: Path):
    """Try UTF-8, fall back to Windows-1252 (Excel default on Windows).
    Yields dict rows."""
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                rows = list(csv.DictReader(f))
            return rows
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "ascii", b"", 0, 1,
        f"Could not decode {path} as utf-8 / utf-8-sig / cp1252 / latin-1.",
    )


def load_rater_csv(path: Path, rater_name: str) -> dict[str, dict]:
    """Read one rater's CSV; return {row_id: parsed_row}."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run sample_validation_set.py first, then have "
            f"{rater_name} fill in their CSV."
        )
    out: dict[str, dict] = {}
    for row in _read_csv_tolerant(path):
        row_id = row["row_id"]
        out[row_id] = {
            "provider":  row["provider"],
            "prompt_id": row["prompt_id"],
            "category":  row["category"],
            "variant":   row["variant"],
            "scores": {
                dim: _to_int_or_none(row.get(f"human_{dim}"))
                for dim in DIMENSIONS
            },
            "branches": {
                branch: _verdict_or_none(row.get(col))
                for branch, col in DAG_BRANCHES.items()
            },
        }
    return out


def load_judge_scores(provider: str, prompt_id: str) -> Optional[dict]:
    """Pull the LLM-judge scores out of the judged file for one response."""
    path = Path(JUDGED_DIR) / provider / f"{prompt_id}.json"
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    judges = rec.get("judges") or {}
    if "_skipped" in judges:
        return None

    def _round_or_none(s):
        if s is None:
            return None
        try:
            v = int(round(float(s)))
        except (TypeError, ValueError):
            return None
        return v if 1 <= v <= 5 else None

    branch_verdicts = (judges.get("adherence") or {}).get("branch_verdicts") or {}
    return {
        "scores": {
            "affordability":  _round_or_none((judges.get("affordability") or {}).get("score")),
            "cultural":       _round_or_none((judges.get("cultural")      or {}).get("score")),
            "feasibility":    _round_or_none((judges.get("feasibility")   or {}).get("score")),
            # health_accuracy isn't a separate LLM judge — skipped
            "health_accuracy": None,
        },
        "branches": {
            branch: ((branch_verdicts.get(branch) or {}).get("verdict") or "").strip().lower() or None
            for branch in DAG_BRANCHES
        },
    }


# =============================================================
# Computation
# =============================================================
def compute_inter_human(rater_a: dict, rater_b: dict) -> dict:
    """Per-dimension kappa between the two humans."""
    out = {}
    common_ids = sorted(set(rater_a) & set(rater_b))

    for dim in DIMENSIONS:
        a = [rater_a[rid]["scores"][dim] for rid in common_ids]
        b = [rater_b[rid]["scores"][dim] for rid in common_ids]
        k = cohens_kappa(a, b)
        n_paired = sum(1 for x, y in zip(a, b) if x is not None and y is not None)
        out[dim] = {"kappa": k, "n": n_paired, "band": kappa_band(k)}

    for branch in DAG_BRANCHES:
        a = [rater_a[rid]["branches"][branch] for rid in common_ids]
        b = [rater_b[rid]["branches"][branch] for rid in common_ids]
        # Drop not_applicable on either side for kappa computation
        # (we report % agreement on those separately).
        a_filt, b_filt = [], []
        for x, y in zip(a, b):
            if x in {"yes", "partial", "no"} and y in {"yes", "partial", "no"}:
                a_filt.append(x); b_filt.append(y)
        k = cohens_kappa(a_filt, b_filt)
        out[f"adh_{branch}"] = {
            "kappa": k,
            "n": len(a_filt),
            "band": kappa_band(k),
        }

    return out


def compute_judge_vs_human(rater_a: dict, rater_b: dict) -> dict:
    """For each LLM judge, kappa against the human CONSENSUS — defined as
    rows where both humans agreed on the score. Rows where humans disagreed
    are dropped (this is the standard treatment when validating against
    a noisy human ground truth)."""
    out = {}
    common_ids = sorted(set(rater_a) & set(rater_b))

    for dim in DIMENSIONS:
        if dim == "health_accuracy":
            # No corresponding LLM judge — skip.
            out[dim] = {"kappa": None, "n": 0, "band": "no LLM judge"}
            continue

        consensus: list[int] = []
        judge: list[int] = []
        for rid in common_ids:
            ha = rater_a[rid]["scores"][dim]
            hb = rater_b[rid]["scores"][dim]
            if ha is None or hb is None or ha != hb:
                continue
            js = load_judge_scores(rater_a[rid]["provider"], rater_a[rid]["prompt_id"])
            if js is None:
                continue
            jscore = js["scores"].get(dim)
            if jscore is None:
                continue
            consensus.append(ha)
            judge.append(jscore)

        k = cohens_kappa(consensus, judge)
        out[dim] = {"kappa": k, "n": len(consensus), "band": kappa_band(k)}

    for branch in DAG_BRANCHES:
        consensus: list[str] = []
        judge: list[str] = []
        for rid in common_ids:
            ha = rater_a[rid]["branches"][branch]
            hb = rater_b[rid]["branches"][branch]
            if ha not in {"yes", "partial", "no"} or hb != ha:
                continue
            js = load_judge_scores(rater_a[rid]["provider"], rater_a[rid]["prompt_id"])
            if js is None:
                continue
            jverdict = js["branches"].get(branch)
            if jverdict not in {"yes", "partial", "no"}:
                continue
            consensus.append(ha)
            judge.append(jverdict)

        k = cohens_kappa(consensus, judge)
        out[f"adh_{branch}"] = {
            "kappa": k, "n": len(consensus), "band": kappa_band(k),
        }

    return out


# =============================================================
# Output
# =============================================================
def write_kappa_csv(inter: dict, judge: dict, path: Path):
    rows = []
    for dim, stats in inter.items():
        rows.append({
            "comparison": "inter_human",
            "dimension":  dim,
            "n_paired":   stats["n"],
            "kappa":      round(stats["kappa"], 3) if stats["kappa"] is not None else "",
            "band":       stats["band"],
        })
    for dim, stats in judge.items():
        rows.append({
            "comparison": "judge_vs_human_consensus",
            "dimension":  dim,
            "n_paired":   stats["n"],
            "kappa":      round(stats["kappa"], 3) if stats["kappa"] is not None else "",
            "band":       stats["band"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["comparison", "dimension", "n_paired", "kappa", "band"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def print_console_summary(inter: dict, judge: dict):
    print("\n" + "=" * 78)
    print("PHASE 6 — COHEN'S KAPPA VALIDATION SUMMARY")
    print("=" * 78)

    def _fmt(stats):
        k = stats["kappa"]
        return f"{k:>+.3f} ({stats['band']})  n={stats['n']:>3}" if k is not None \
            else f"  n/a               n={stats['n']:>3}"

    print("\nInter-human agreement (upper bound on any LLM judge):")
    for dim, stats in inter.items():
        print(f"  {dim:32s}  {_fmt(stats)}")

    print("\nLLM judge vs human consensus (only rows where humans agreed):")
    for dim, stats in judge.items():
        print(f"  {dim:32s}  {_fmt(stats)}")

    print("\nInterpretation (Landis & Koch 1977):")
    print("  < 0.00  worse than chance     0.40-0.60  moderate")
    print("  0.00-0.20  slight             0.60-0.80  substantial")
    print("  0.20-0.40  fair               0.80-1.00  almost perfect")


# =============================================================
# Main
# =============================================================
def main():
    parser = argparse.ArgumentParser(description="Cohen's kappa validation")
    parser.add_argument("--rater-names", nargs=2, default=["sanjana", "snigdha"],
                        help="Names of the two raters (matches CSV filenames)")
    parser.add_argument("--out", default=str(Path(RESULTS_DIR) / "kappa_report.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    a_path = VALIDATION_DIR / f"human_scores_{args.rater_names[0]}.csv"
    b_path = VALIDATION_DIR / f"human_scores_{args.rater_names[1]}.csv"

    print(f"Loading rater A: {a_path}")
    rater_a = load_rater_csv(a_path, args.rater_names[0])
    print(f"  {len(rater_a)} rows")

    print(f"Loading rater B: {b_path}")
    rater_b = load_rater_csv(b_path, args.rater_names[1])
    print(f"  {len(rater_b)} rows")

    common = set(rater_a) & set(rater_b)
    print(f"  {len(common)} rows in common (used for kappa)")

    inter = compute_inter_human(rater_a, rater_b)
    judge = compute_judge_vs_human(rater_a, rater_b)
    write_kappa_csv(inter, judge, Path(args.out))
    print_console_summary(inter, judge)


if __name__ == "__main__":
    main()
